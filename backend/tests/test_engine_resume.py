"""B6 — engine resume/crash integration tests on real Postgres.

These are the phase-B acceptance gates (improvement-plan §B):
* kill the worker mid side_effect tool → resume drops to waiting_input and does **not**
  re-fire the side effect (at-most-once);
* kill the worker mid LLM step → resume reconstructs the conversation from run_steps and
  continues from the last step (replay-safe), with no duplicate step;
* quota on the line → the second run fails `quota_exceeded` and users.used == Σ run_steps.tokens;
* a task opened mid-flight → snapshot reconstructs the full worklog timeline.

A real worker crash is simulated with `_Crash(BaseException)`: the runner's tool/LLM guards
are `except Exception`, so a BaseException escapes them exactly like a process dying mid-step,
leaving the half-written state (a `pending` tool step, or no step at all) for resume to settle.

The runner is driven directly against a fresh engine (the local-engine technique — dodges the
module-engine event-loop issue) with `db_factory` + injected provider/tools. Redis/event I/O is
patched out so these tests isolate the *Postgres* correctness B6 is about; the event stream is
covered in test_task_stream + the B5 pubsub smoke.

    docker compose exec backend pytest tests/test_engine_resume.py
"""
from __future__ import annotations

import asyncio
import uuid

from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import settings
from app.plugins.ai.models import Agent, Task, Run
from app.plugins.auth.models import User
from app.plugins.ai import runs_repo
from app.plugins.ai import stub_tools_repo as stub_repo
from app.plugins.auth import users_repo
from app.plugins.ai import agent_runner, task_service
from app.plugins.ai.agent_runner import EFFECT_READ, EFFECT_SIDE_EFFECT, LLMResult


class _Crash(BaseException):
    """Stands in for the worker process dying — escapes the runner's `except Exception`."""


# --- injected collaborators -------------------------------------------------


class IntProvider:
    """Turn-keyed scripted provider (turn = #assistant messages, so it's resume-stable).
    `crash_on_turn` raises _Crash the *first* time that turn is reached (then succeeds)."""

    def __init__(self, script, crash_on_turn=None):
        self.script = script
        self.crash_on_turn = crash_on_turn
        self._crashed = set()
        self.last_messages = None

    async def complete(self, *, model, messages, tools):
        self.last_messages = list(messages)
        turn = sum(1 for m in messages if m.get("role") == "assistant")
        if turn == self.crash_on_turn and turn not in self._crashed:
            self._crashed.add(turn)
            raise _Crash(f"died mid-LLM (turn {turn})")
        spec = self.script[turn] if turn < len(self.script) else {"text": "(end)", "tokens": 1}
        text, tokens = str(spec.get("text", "")), int(spec.get("tokens", 1))
        if spec.get("tool"):
            return LLMResult(text=text, stop_reason="tool_use", tokens=tokens,
                             tool_name=spec["tool"], tool_args=spec.get("args", {}))
        return LLMResult(text=text, stop_reason="end", tokens=tokens)


class ITool:
    """Records every dispatch and applies its side effect (a stub_tool_writes row) before
    optionally crashing — so the at-most-once guarantee is checked against real rows."""

    def __init__(self, Session, effects, crash_on=None):
        self.Session = Session
        self.effects = effects
        self.crash_on = crash_on
        self.calls = []

    def schemas(self):
        return []

    def effect_of(self, name):
        return self.effects.get(name, EFFECT_SIDE_EFFECT)

    async def call(self, name, args, *, idempotency_key):
        self.calls.append((name, idempotency_key))
        rid = uuid.UUID(idempotency_key.split(":", 1)[0])
        async with self.Session() as db:                      # the side effect fires here
            await stub_repo.record_write(db, rid, name, idempotency_key, args)
        if name == self.crash_on:
            raise _Crash(f"died after side-effect {name}")     # ...then the worker dies
        return {"ok": True, "key": idempotency_key}


# --- harness ----------------------------------------------------------------


async def _never_cancelled(_rid):
    return False


async def _noop(*a, **k):
    return None


def _patch_io(monkeypatch):
    """Isolate to Postgres: no Redis cancel checks, no event publishing in these tests."""
    monkeypatch.setattr(agent_runner.redis_bus, "is_run_cancelled", _never_cancelled)
    monkeypatch.setattr(agent_runner.events, "publish_step", _noop)
    monkeypatch.setattr(agent_runner.events, "publish_run", _noop)


def _run(coro):
    async def main():
        eng = create_async_engine(settings.database_url)
        Session = async_sessionmaker(eng, expire_on_commit=False, class_=AsyncSession)
        try:
            return await coro(Session)
        finally:
            await eng.dispose()

    return asyncio.run(main())


async def _make_run(Session, *, task_id=None, agent_id=None) -> uuid.UUID:
    rid = uuid.uuid4()
    async with Session() as db:
        db.add(Run(id=rid, task_id=task_id, agent_id=agent_id, status="queued",
                   input={"messages": [{"role": "user", "content": "go"}]}))
        await db.commit()
    return rid


async def _cleanup(Session, *, runs=(), agents=(), tasks=(), users=()):
    async with Session() as db:                                # deleting a run cascades steps + stub writes
        for r in runs:
            await db.execute(delete(Run).where(Run.id == r))
        for a in agents:
            await db.execute(delete(Agent).where(Agent.id == a))
        for q in tasks:
            await db.execute(delete(Task).where(Task.id == q))
        for u in users:
            await db.execute(delete(User).where(User.id == u))
        await db.commit()


# --- tests ------------------------------------------------------------------


def test_kill_mid_side_effect_resumes_waiting_no_double_write(monkeypatch):
    _patch_io(monkeypatch)

    async def body(Session):
        rid = await _make_run(Session)
        provider = IntProvider([{"tool": "pay", "tokens": 3}, {"text": "done", "tokens": 2}])
        tools = ITool(Session, {"pay": EFFECT_SIDE_EFFECT}, crash_on="pay")

        try:                                                    # run #1 — crashes mid side-effect
            await agent_runner.run(rid, provider=provider, tools=tools, db_factory=Session)
            crashed = False
        except _Crash:
            crashed = True
        async with Session() as db:
            after_crash = [(s.kind, s.status) for s in await runs_repo.list_steps(db, rid)]
            writes1 = await stub_repo.count_writes(db, rid)

        status = await agent_runner.run(rid, provider=provider, tools=tools, db_factory=Session)  # resume
        async with Session() as db:
            run2 = await runs_repo.get_run(db, rid)
            writes2 = await stub_repo.count_writes(db, rid)
        await _cleanup(Session, runs=[rid])
        return crashed, after_crash, writes1, status, run2.status, writes2, tools.calls

    crashed, after_crash, w1, status, run2status, w2, calls = _run(body)
    assert crashed
    assert after_crash == [("llm", "done"), ("tool", "pending")]   # half-written: pending tool
    assert w1 == 1
    assert status == "waiting_input" and run2status == "waiting_input"
    assert w2 == 1 and len(calls) == 1                              # side effect NOT re-fired


def test_kill_mid_llm_resumes_same_conversation(monkeypatch):
    _patch_io(monkeypatch)

    async def body(Session):
        rid = await _make_run(Session)
        provider = IntProvider([{"tool": "echo", "tokens": 4}, {"text": "final", "tokens": 2}], crash_on_turn=1)
        tools = ITool(Session, {"echo": EFFECT_READ})

        try:                                                    # run #1 — crashes mid 2nd LLM
            await agent_runner.run(rid, provider=provider, tools=tools, db_factory=Session)
            crashed = False
        except _Crash:
            crashed = True
        async with Session() as db:
            after_crash = [(s.kind, s.status) for s in await runs_repo.list_steps(db, rid)]

        status = await agent_runner.run(rid, provider=provider, tools=tools, db_factory=Session)  # resume
        async with Session() as db:
            final_steps = [(s.kind, s.status) for s in await runs_repo.list_steps(db, rid)]
            run2 = await runs_repo.get_run(db, rid)
        roles = [m.get("role") for m in (provider.last_messages or [])]
        await _cleanup(Session, runs=[rid])
        return crashed, after_crash, status, run2.status, final_steps, roles

    crashed, after_crash, status, run2status, final_steps, roles = _run(body)
    assert crashed
    assert after_crash == [("llm", "done"), ("tool", "done")]       # nothing persisted for the crashed turn
    assert status == "done" and run2status == "done"
    assert final_steps == [("llm", "done"), ("tool", "done"), ("llm", "done")]  # no duplicate
    assert "tool" in roles                                          # resume fed the tool result back to the LLM


def test_quota_on_the_line_used_equals_sum_of_step_tokens(monkeypatch):
    _patch_io(monkeypatch)

    async def body(Session):
        uid, aid = uuid.uuid4(), uuid.uuid4()
        async with Session() as db:
            db.add(User(id=uid, username=f"q_{uid.hex[:8]}", email=f"{uid.hex[:8]}@t.local",
                        display="t", role="member", password_hash="x", quota=6, used=0))
            await db.flush()
            db.add(Agent(id=aid, owner_id=uid, name="a"))
            await db.commit()
        r1 = await _make_run(Session, agent_id=aid)
        r2 = await _make_run(Session, agent_id=aid)
        s1 = await agent_runner.run(r1, provider=IntProvider([{"text": "x", "tokens": 4}]), tools=ITool(Session, {}), db_factory=Session)
        s2 = await agent_runner.run(r2, provider=IntProvider([{"text": "y", "tokens": 4}]), tools=ITool(Session, {}), db_factory=Session)
        async with Session() as db:
            user = await users_repo.get_by_id(db, uid)
            t1 = await runs_repo.total_step_tokens(db, r1)
            t2 = await runs_repo.total_step_tokens(db, r2)
            err = (await runs_repo.get_run(db, r2)).error
            used = user.used
        await _cleanup(Session, runs=[r1, r2], agents=[aid], users=[uid])
        return s1, s2, used, t1, t2, err

    s1, s2, used, t1, t2, err = _run(body)
    assert s1 == "done" and s2 == "failed"
    assert err == "quota_exceeded"
    assert used == t1 + t2 == 4   # only the successful step's tokens reserved; rejected step persists 0


def test_snapshot_reconstructs_full_timeline(monkeypatch):
    _patch_io(monkeypatch)

    async def body(Session):
        qid = uuid.uuid4()
        async with Session() as db:
            db.add(Task(id=qid, title="t", status="open"))
            await db.commit()
        rid = await _make_run(Session, task_id=qid)
        provider = IntProvider([{"tool": "echo", "tokens": 3}, {"text": "done", "tokens": 2}])
        await agent_runner.run(rid, provider=provider, tools=ITool(Session, {"echo": EFFECT_READ}), db_factory=Session)
        async with Session() as db:
            snap = await task_service.snapshot(db, str(qid))
        await _cleanup(Session, runs=[rid], tasks=[qid])
        return snap

    snap = _run(body)
    assert [s["kind"] for s in snap["steps"]] == ["llm", "tool", "llm"]   # full timeline recovered
    assert snap["runs"] and snap["runs"][0]["status"] == "done"
