"""Agent execution engine — the `run(run_id)` loop run by the arq worker (B3).

One agent run = an agent executing one task via a loop of LLM calls + tool calls
(system-design §4). This module owns the *control* logic; all SQL is in
`repositories/runs.py`, and the LLM provider + tool registry are **injected** (the real
stub provider/tool land in B4, real adapters in C1). That keeps the loop provider-agnostic
and the engine's correctness (resume, quota, timeouts) testable in isolation.

Correctness guarantees (risk-mitigation §1, arq is at-least-once → every step replay-safe):

* **Two-phase tool steps** — a `tool` step is persisted `pending` with a deterministic
  `idempotency_key = "{run_id}:{seq}"` *before* dispatch, then flipped `done` with the
  result. A crash between the two leaves a visible `pending` step.
* **Resume by effect class** — on restart a trailing `pending` tool step is decided by its
  effect: `read`/`idempotent_write` → re-run with the same key; `side_effect` → the run
  drops to `waiting_input` (never auto-retried — at-most-once for money/messages).
* **Atomic quota** — LLM tokens are reserved with one conditional UPDATE before the step is
  persisted, so `users.used` always equals Σ run_steps.tokens and concurrent runs can't race
  past the ceiling.
* **Bounds** — per-step timeouts (`run_llm_step_timeout_s`/`run_tool_step_timeout_s`),
  `run_max_steps`, `run_max_wallclock_s`, and a Redis cancel flag checked between steps.
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from . import redis_bus
from ...core.config import settings
from ...core.contracts import Retriever  # the RAG contract lives in the kernel now (ai consumes it)
from ...core.logging_ctx import bind_run, reset_run
from . import db_ref
from . import runs_repo
from . import events

log = logging.getLogger("pikaos.engine")

# --- effect classes (tools_config.config.effect — risk-mitigation §1) ---
EFFECT_READ = "read"
EFFECT_IDEMPOTENT_WRITE = "idempotent_write"
EFFECT_SIDE_EFFECT = "side_effect"
# Unknown/unclassified tools are treated as side_effect — the safe default: never
# auto-retry something we can't prove is replay-safe.
_REPLAYABLE = {EFFECT_READ, EFFECT_IDEMPOTENT_WRITE}

STATUS_PENDING = "pending"
STATUS_DONE = "done"
STATUS_FAILED = "failed"


# --- injected collaborators (real implementations: B4 stubs, C1 adapters) ---


@dataclass
class LLMResult:
    """Normalized output of one LLM call (vendor tool-use flattened — system-design §4)."""

    text: str = ""
    stop_reason: str = "end"          # "end" (final answer) | "tool_use"
    tokens: int = 0
    tool_name: str | None = None
    tool_args: dict = field(default_factory=dict)


class LLMProvider(Protocol):
    async def complete(self, *, model: str, messages: list[dict], tools: list[dict]) -> LLMResult: ...


class ToolRegistry(Protocol):
    def schemas(self) -> list[dict]: ...
    def effect_of(self, name: str) -> str: ...
    async def call(self, name: str, args: dict, *, idempotency_key: str) -> dict: ...


# --- pure helpers (no DB/Redis — unit-tested directly, like rbac_service.resolve_perms) ---


def classify_effect(effect: str | None) -> str:
    """Normalize a tool's declared effect; anything unrecognized → side_effect (safe default)."""
    return effect if effect in (EFFECT_READ, EFFECT_IDEMPOTENT_WRITE, EFFECT_SIDE_EFFECT) else EFFECT_SIDE_EFFECT


def resume_action(last_status: str | None, effect: str | None) -> str:
    """Decide what to do with the trailing step when resuming a `running` run.

    Returns one of: "continue" (no unfinished step / last step done) ·
    "rerun" (pending tool, replay-safe effect → re-dispatch with the same key) ·
    "wait_input" (pending side_effect → hand to a human, never auto-retry).
    """
    if last_status != STATUS_PENDING:
        return "continue"
    return "rerun" if classify_effect(effect) in _REPLAYABLE else "wait_input"


def messages_from_run(run_input: dict | None, steps: list) -> list[dict]:
    """Reconstruct the LLM conversation from the run seed + persisted `done` steps.

    Seed = run.input["messages"] verbatim, else a single user turn from run.input["task"].
    Replays only completed llm/tool steps (a trailing pending step is handled by the
    resume logic before this is called), so the provider sees the same context it left off at.
    """
    run_input = run_input or {}
    seed = run_input.get("messages")
    messages: list[dict] = list(seed) if isinstance(seed, list) else []
    if not messages and run_input.get("task"):
        messages.append({"role": "user", "content": str(run_input["task"])})

    for s in steps:
        if s.status != STATUS_DONE:
            continue
        content = s.content or {}
        if s.kind == "llm":
            msg: dict = {"role": "assistant", "content": content.get("text", "")}
            if content.get("tool_name"):
                msg["tool_call"] = {"name": content["tool_name"], "args": content.get("tool_args", {})}
            messages.append(msg)
        elif s.kind == "tool":
            intent = content.get("intent", {})
            messages.append({"role": "tool", "name": intent.get("name", ""), "content": content.get("result")})
    return messages


# --- the run loop (DB I/O) ---


async def run(
    run_id: uuid.UUID | str,
    *,
    provider: LLMProvider,
    tools: ToolRegistry,
    retriever: "Retriever | None" = None,
    db_factory=db_ref.new_session,
) -> str:
    """Execute (or resume) one agent run to a terminal status. Returns the final status.

    Safe to call again on the same run after a crash: it reconstructs state from
    `run_steps` and continues. Idempotent on an already-terminal run.
    """
    rid = uuid.UUID(str(run_id))
    bind_run(run_id=str(rid))  # so every log line below carries the run (B7)
    async with db_factory() as db:
        run_row = await runs_repo.get_run(db, rid)
        if run_row is None:
            log.warning("agent_run: run %s not found", rid)
            return "missing"
        if run_row.status in ("done", "failed", "cancelled"):
            return run_row.status  # already terminal — replay no-op

        agent = await runs_repo.get_agent(db, run_row.agent_id) if run_row.agent_id else None
        owner_id = agent.owner_id if agent else None
        model = (agent.model if agent and agent.model else "") or "stub"
        # task the worklog streams to (None → run not bound to a task → events go nowhere)
        task_id = str(run_row.task_id) if run_row.task_id else None
        # enrich the log context now that the run is loaded (B7)
        bind_run(
            parent_run_id=str(run_row.parent_run_id) if run_row.parent_run_id else None,
            task_id=task_id,
            agent_id=str(run_row.agent_id) if run_row.agent_id else None,
        )

        steps = await runs_repo.list_steps(db, rid)
        seq = (steps[-1].seq + 1) if steps else 0

        # --- resume: settle a trailing pending tool step before doing anything else ---
        if steps and steps[-1].status == STATUS_PENDING:
            last = steps[-1]
            effect = classify_effect((last.content or {}).get("effect"))
            action = resume_action(last.status, effect)
            if action == "wait_input":
                await runs_repo.set_run_status(db, rid, "waiting_input")
                await runs_repo.set_agent_status(db, run_row.agent_id, "idle")
                await events.publish_run(task_id, rid, "waiting_input")
                log.info("run %s resumed onto pending side_effect → waiting_input", rid)
                return "waiting_input"
            # replay-safe → re-dispatch with the SAME idempotency_key, then complete the step
            intent = (last.content or {}).get("intent", {})
            try:
                result = await asyncio.wait_for(
                    tools.call(intent.get("name", ""), intent.get("args", {}), idempotency_key=last.idempotency_key or ""),
                    timeout=settings.run_tool_step_timeout_s,
                )
            except (asyncio.TimeoutError, Exception) as exc:  # noqa: BLE001 — any tool failure fails the run
                await runs_repo.complete_step(db, last.id, status=STATUS_FAILED, content={**(last.content or {}), "error": str(exc)})
                return await _fail(db, run_row, f"tool_error: {exc}")
            await runs_repo.complete_step(db, last.id, status=STATUS_DONE, content={**(last.content or {}), "result": result})
            last.status, last.content = STATUS_DONE, {**(last.content or {}), "result": result}
            await events.publish_step(task_id, last)
            steps = await runs_repo.list_steps(db, rid)
            seq = steps[-1].seq + 1

        await runs_repo.set_run_status(db, rid, "running", started=(run_row.started_at is None))
        await runs_repo.set_agent_status(db, run_row.agent_id, "busy")
        await events.publish_run(task_id, rid, "running")

        messages = messages_from_run(run_row.input, steps)
        # RAG (E3): prepend top-k codex context, scoped to the run owner. Off unless configured
        # (engine_retrieval_top_k > 0). Side-effect-free (no step, no quota) → safe to re-derive on
        # resume; failures must never sink a run, so retrieval is best-effort.
        if retriever is not None and settings.engine_retrieval_top_k > 0:
            try:
                ctx = await retriever.retrieve_context(
                    db, owner_id=owner_id, run_input=run_row.input, k=settings.engine_retrieval_top_k,
                )
                if ctx:
                    messages.insert(0, {"role": "system", "content": ctx})
            except Exception:  # noqa: BLE001 — retrieval is an enhancement, never a failure mode
                log.exception("run %s: retrieval failed, continuing without context", rid)
        # steps already executed this run (caps apply to the whole run, across resumes)
        used_steps = sum(1 for s in steps if s.kind in ("llm", "tool"))
        started = time.monotonic()
        tool_schemas = tools.schemas()
        log.info("run started (kind=%s, model=%s, resumed_steps=%d)", run_row.kind, model, used_steps)

        while True:
            if await redis_bus.is_run_cancelled(str(rid)):
                await runs_repo.set_run_status(db, rid, "cancelled", ended=True)
                await runs_repo.set_agent_status(db, run_row.agent_id, "idle")
                await events.publish_run(task_id, rid, "cancelled")
                return "cancelled"
            if used_steps >= settings.run_max_steps:
                return await _fail(db, run_row, "max_steps_exceeded")
            if time.monotonic() - started > settings.run_max_wallclock_s:
                return await _fail(db, run_row, "wallclock_exceeded")

            # --- LLM step ---
            try:
                llm = await asyncio.wait_for(
                    provider.complete(model=model, messages=messages, tools=tool_schemas),
                    timeout=settings.run_llm_step_timeout_s,
                )
            except asyncio.TimeoutError:
                return await _fail(db, run_row, "llm_timeout")
            except Exception as exc:  # noqa: BLE001
                return await _fail(db, run_row, f"llm_error: {exc}")

            # Reserve quota BEFORE persisting the step → users.used stays == Σ run_steps.tokens.
            if not await runs_repo.reserve_quota(db, owner_id, llm.tokens):
                qstep = await runs_repo.insert_step(
                    db, rid, seq, "status", content={"error": "quota_exceeded", "needed": llm.tokens}, tokens=0,
                )
                await events.publish_step(task_id, qstep)
                return await _fail(db, run_row, "quota_exceeded")

            step = await runs_repo.insert_step(
                db, rid, seq, "llm", role="assistant", tokens=llm.tokens,
                content={"text": llm.text, "tool_name": llm.tool_name, "tool_args": llm.tool_args},
            )
            await runs_repo.add_run_tokens(db, rid, llm.tokens)
            await events.publish_step(task_id, step)
            seq += 1
            used_steps += 1
            asst: dict = {"role": "assistant", "content": llm.text}
            if llm.tool_name:
                asst["tool_call"] = {"name": llm.tool_name, "args": llm.tool_args}
            messages.append(asst)

            if llm.stop_reason != "tool_use" or not llm.tool_name:
                await runs_repo.set_run_status(db, rid, "done", ended=True)
                await runs_repo.set_agent_status(db, run_row.agent_id, "idle")
                await events.publish_run(task_id, rid, "done", tokens_used=run_row.tokens_used)
                log.info("run done (steps=%d, tokens=%d)", used_steps, run_row.tokens_used)
                return "done"

            # --- two-phase tool step ---
            name, targs = llm.tool_name, llm.tool_args
            effect = classify_effect(tools.effect_of(name))
            idem = f"{rid}:{seq}"
            pending = await runs_repo.insert_step(
                db, rid, seq, "tool", status=STATUS_PENDING, idempotency_key=idem, role="tool",
                content={"intent": {"name": name, "args": targs}, "effect": effect},
            )
            await events.publish_step(task_id, pending)  # pending shows in the timeline immediately
            seq += 1
            used_steps += 1
            try:
                result = await asyncio.wait_for(
                    tools.call(name, targs, idempotency_key=idem),
                    timeout=settings.run_tool_step_timeout_s,
                )
            except asyncio.TimeoutError:
                await runs_repo.complete_step(db, pending.id, status=STATUS_FAILED, content={**(pending.content or {}), "error": "tool_timeout"})
                pending.status, pending.content = STATUS_FAILED, {**(pending.content or {}), "error": "tool_timeout"}
                await events.publish_step(task_id, pending)
                return await _fail(db, run_row, "tool_timeout")
            except Exception as exc:  # noqa: BLE001
                await runs_repo.complete_step(db, pending.id, status=STATUS_FAILED, content={**(pending.content or {}), "error": str(exc)})
                pending.status, pending.content = STATUS_FAILED, {**(pending.content or {}), "error": str(exc)}
                await events.publish_step(task_id, pending)
                return await _fail(db, run_row, f"tool_error: {exc}")

            await runs_repo.complete_step(db, pending.id, status=STATUS_DONE, content={**(pending.content or {}), "result": result})
            pending.status, pending.content = STATUS_DONE, {**(pending.content or {}), "result": result}
            await events.publish_step(task_id, pending)
            messages.append({"role": "tool", "name": name, "content": result})
            # loop back: feed the tool result to the next LLM step


# --- worker entrypoint glue ---
# The provider + tool registry are configured once at worker startup (B4 stubs / C1 adapters)
# via set_engine_runtime, so the arq `agent_run` job stays a thin shim over run().
_runtime: tuple[LLMProvider, ToolRegistry, "Retriever | None"] | None = None


def set_engine_runtime(
    provider: LLMProvider, tools: ToolRegistry, retriever: "Retriever | None" = None
) -> None:
    """Wire the engine's runtime deps once at worker startup. `retriever` is optional and injected by
    the composition root (worker) only when the knowledge plugin is active — so the engine stays
    decoupled from knowledge (modularity §2)."""
    global _runtime
    _runtime = (provider, tools, retriever)


async def run_job(run_id: str) -> str:
    """arq `agent_run` job — resolves the configured runtime and runs one agent run.

    Binds the run's log context for the whole job and resets it on exit, so log lines are
    attributed to this run and nothing leaks into the next job on the worker (B7)."""
    if _runtime is None:
        raise RuntimeError("engine runtime not configured — call set_engine_runtime() at worker startup (B4)")
    provider, tools, retriever = _runtime
    token = bind_run(run_id=str(run_id))
    try:
        return await run(run_id, provider=provider, tools=tools, retriever=retriever)
    finally:
        reset_run(token)


async def _fail(db, run_row, error: str) -> str:
    """Mark the run failed + free its agent, emit the terminal event, return 'failed'."""
    await runs_repo.set_run_status(db, run_row.id, "failed", ended=True, error=error)
    await runs_repo.set_agent_status(db, run_row.agent_id, "idle")
    task_id = str(run_row.task_id) if run_row.task_id else None
    await events.publish_run(task_id, run_row.id, "failed", error=error)
    log.info("run %s failed: %s", run_row.id, error)
    return "failed"
