"""Tests for the agent execution engine (B3) — services/agent_runner.

Two layers, mirroring test_rbac.py's "pure first, no live server" style:

* pure helpers (classify_effect / resume_action / messages_from_run) tested directly;
* the run() loop driven against an in-memory fake of repositories/runs + redis_bus,
  with a scripted fake LLM provider + fake tool registry — so resume, two-phase tools,
  atomic quota, timeouts and cancel are asserted with no DB. (The real-Postgres harness
  that kills a worker mid-step is B6.)

    docker compose exec backend pytest tests/test_agent_runner.py
"""
from __future__ import annotations

import asyncio
import uuid
from types import SimpleNamespace

import pytest

from app.core.config import settings
from app.plugins.ai import agent_runner
from app.plugins.ai.agent_runner import (
    EFFECT_READ,
    EFFECT_SIDE_EFFECT,
    LLMResult,
    classify_effect,
    messages_from_run,
    resume_action,
)


# --- pure helpers -----------------------------------------------------------


def test_classify_effect_defaults_unknown_to_side_effect():
    assert classify_effect("read") == "read"
    assert classify_effect("idempotent_write") == "idempotent_write"
    assert classify_effect("side_effect") == "side_effect"
    assert classify_effect(None) == "side_effect"
    assert classify_effect("garbage") == "side_effect"


def test_resume_action():
    assert resume_action("done", "side_effect") == "continue"   # nothing pending
    assert resume_action(None, None) == "continue"
    assert resume_action("pending", "read") == "rerun"          # replay-safe
    assert resume_action("pending", "idempotent_write") == "rerun"
    assert resume_action("pending", "side_effect") == "wait_input"
    assert resume_action("pending", None) == "wait_input"        # unknown → never auto-retry


def test_messages_from_run_seed_and_replay():
    steps = [
        SimpleNamespace(kind="llm", status="done", content={"text": "thinking", "tool_name": "echo", "tool_args": {"x": 1}}),
        SimpleNamespace(kind="tool", status="done", content={"intent": {"name": "echo"}, "result": {"ok": True}}),
        SimpleNamespace(kind="llm", status="pending", content={"text": "ignored"}),  # not replayed
    ]
    msgs = messages_from_run({"task": "do it"}, steps)
    assert msgs[0] == {"role": "user", "content": "do it"}
    assert msgs[1]["role"] == "assistant" and msgs[1]["tool_call"] == {"name": "echo", "args": {"x": 1}}
    assert msgs[2] == {"role": "tool", "name": "echo", "content": {"ok": True}}
    assert len(msgs) == 3  # the pending step is skipped


# --- in-memory fakes for the loop -------------------------------------------


class FakeStore:
    """Stands in for repositories/runs — an in-memory runs/steps/quota model."""

    def __init__(self, run, agent=None, quota=None, used=0):
        self.run = run
        self.agent = agent
        self.quota = quota
        self.used = used
        self.steps: list = []

    # repo surface used by agent_runner.run
    async def get_run(self, db, rid):
        return self.run

    async def get_agent(self, db, aid):
        return self.agent

    async def list_steps(self, db, rid):
        return list(self.steps)

    async def set_run_status(self, db, rid, status, *, started=False, ended=False, error=None):
        self.run.status = status
        if error is not None:
            self.run.error = error
        if started:
            self.run.started_at = "now"

    async def set_agent_status(self, db, aid, status):
        if self.agent is not None:
            self.agent.status = status

    async def insert_step(self, db, rid, seq, kind, *, status="done", idempotency_key=None, role=None, content=None, tokens=0):
        step = SimpleNamespace(
            id=uuid.uuid4(), run_id=rid, seq=seq, kind=kind, status=status,
            idempotency_key=idempotency_key, role=role, content=content, tokens=tokens,
        )
        self.steps.append(step)
        return step

    async def complete_step(self, db, step_id, *, status="done", content=None, tokens=0):
        for s in self.steps:
            if s.id == step_id:
                s.status = status
                if content is not None:
                    s.content = content
                s.tokens = tokens

    async def add_run_tokens(self, db, run_id, tokens):
        if tokens > 0:
            self.run.tokens_used += tokens

    async def reserve_quota(self, db, owner_id, tokens):
        if owner_id is None or tokens <= 0:
            return True
        if self.quota is not None and self.used + tokens > self.quota:
            return False
        self.used += tokens
        return True

    def total_tokens(self):
        return sum(s.tokens for s in self.steps)


class FakeProvider:
    """Scripted LLM — returns the next LLMResult per complete() call."""

    def __init__(self, script, sleep=0.0):
        self.script = list(script)
        self.sleep = sleep
        self.calls = 0

    async def complete(self, *, model, messages, tools):
        self.calls += 1
        if self.sleep:
            await asyncio.sleep(self.sleep)
        return self.script.pop(0)


class FakeTools:
    def __init__(self, effects, behavior=None):
        self.effects = effects
        self.behavior = behavior or {}
        self.calls = []

    def schemas(self):
        return []

    def effect_of(self, name):
        return self.effects.get(name, EFFECT_SIDE_EFFECT)

    async def call(self, name, args, *, idempotency_key):
        self.calls.append((name, args, idempotency_key))
        beh = self.behavior.get(name)
        if beh == "raise":
            raise RuntimeError("boom")
        if isinstance(beh, (int, float)) and beh:
            await asyncio.sleep(beh)
        return {"echo": args, "key": idempotency_key}


class _FakeDB:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


def _patch(monkeypatch, store):
    monkeypatch.setattr(agent_runner, "runs_repo", store)
    monkeypatch.setattr(agent_runner.redis_bus, "is_run_cancelled", _never_cancelled)


async def _never_cancelled(rid):
    return False


def _run(store, provider, tools):
    return asyncio.run(agent_runner.run(store.run.id, provider=provider, tools=tools, db_factory=lambda: _FakeDB()))


def _mk(quota=None, used=0, owner=True, started_at=None):
    aid = uuid.uuid4()
    run = SimpleNamespace(id=uuid.uuid4(), status="queued", kind="agent", agent_id=aid, task_id=None, parent_run_id=None, started_at=started_at, input={"task": "go"}, error=None, tokens_used=0)
    agent = SimpleNamespace(id=aid, owner_id=(uuid.uuid4() if owner else None), model="stub", status="idle")
    return FakeStore(run, agent, quota=quota, used=used)


# --- the loop ---------------------------------------------------------------


def test_simple_run_completes(monkeypatch):
    store = _mk()
    _patch(monkeypatch, store)
    provider = FakeProvider([LLMResult(text="all done", stop_reason="end", tokens=5)])
    out = _run(store, provider, FakeTools({}))
    assert out == "done"
    assert store.run.status == "done"
    assert store.agent.status == "idle"
    assert store.used == 5 == store.total_tokens()  # user quota == Σ step tokens
    assert store.run.tokens_used == 5               # run total rolled up too


def test_two_phase_tool_then_finish(monkeypatch):
    store = _mk()
    _patch(monkeypatch, store)
    provider = FakeProvider([
        LLMResult(text="use tool", stop_reason="tool_use", tokens=3, tool_name="echo", tool_args={"a": 1}),
        LLMResult(text="done", stop_reason="end", tokens=2),
    ])
    tools = FakeTools({"echo": EFFECT_READ})
    out = _run(store, provider, tools)
    assert out == "done"
    kinds = [(s.kind, s.status) for s in store.steps]
    assert kinds == [("llm", "done"), ("tool", "done"), ("llm", "done")]
    # tool step got a deterministic idempotency_key and was dispatched once
    tool_step = store.steps[1]
    assert tool_step.idempotency_key == f"{store.run.id}:1"
    assert tools.calls == [("echo", {"a": 1}, f"{store.run.id}:1")]
    assert tool_step.content["result"] == {"echo": {"a": 1}, "key": f"{store.run.id}:1"}


def test_quota_exceeded_fails_and_used_matches_steps(monkeypatch):
    # quota fits the first step (4) but not the second (4+4 > 6)
    store = _mk(quota=6)
    _patch(monkeypatch, store)
    provider = FakeProvider([
        LLMResult(text="step1", stop_reason="tool_use", tokens=4, tool_name="echo", tool_args={}),
        LLMResult(text="step2", stop_reason="end", tokens=4),
    ])
    out = _run(store, provider, FakeTools({"echo": EFFECT_READ}))
    assert out == "failed"
    assert store.run.error == "quota_exceeded"
    # used reflects only the reserved (persisted-token) steps; the rejected step persists 0 tokens
    assert store.used == 4 == store.total_tokens()


def test_resume_pending_read_tool_reruns_same_key(monkeypatch):
    store = _mk(started_at="earlier")
    store.run.status = "running"
    rid = store.run.id
    # a tool step left pending by a crash (effect=read → replay-safe)
    store.steps.append(SimpleNamespace(
        id=uuid.uuid4(), run_id=rid, seq=0, kind="tool", status="pending",
        idempotency_key=f"{rid}:0", role="tool",
        content={"intent": {"name": "echo", "args": {"k": 9}}, "effect": EFFECT_READ}, tokens=0,
    ))
    _patch(monkeypatch, store)
    provider = FakeProvider([LLMResult(text="wrap up", stop_reason="end", tokens=1)])
    tools = FakeTools({"echo": EFFECT_READ})
    out = _run(store, provider, tools)
    assert out == "done"
    # the pending step was re-dispatched with the SAME key, then completed
    assert tools.calls == [("echo", {"k": 9}, f"{rid}:0")]
    assert store.steps[0].status == "done"
    assert store.steps[0].content["result"]["key"] == f"{rid}:0"


def test_resume_pending_side_effect_waits_for_human(monkeypatch):
    store = _mk(started_at="earlier")
    store.run.status = "running"
    rid = store.run.id
    store.steps.append(SimpleNamespace(
        id=uuid.uuid4(), run_id=rid, seq=0, kind="tool", status="pending",
        idempotency_key=f"{rid}:0", role="tool",
        content={"intent": {"name": "pay", "args": {}}, "effect": EFFECT_SIDE_EFFECT}, tokens=0,
    ))
    _patch(monkeypatch, store)
    tools = FakeTools({"pay": EFFECT_SIDE_EFFECT})
    out = _run(store, FakeProvider([]), tools)
    assert out == "waiting_input"
    assert store.run.status == "waiting_input"
    assert store.agent.status == "idle"
    assert tools.calls == []  # never auto-retried — at-most-once


def test_llm_timeout_fails_run(monkeypatch):
    store = _mk()
    _patch(monkeypatch, store)
    monkeypatch.setattr(settings, "run_llm_step_timeout_s", 0.05)
    provider = FakeProvider([LLMResult(text="late", stop_reason="end", tokens=1)], sleep=0.3)
    out = _run(store, provider, FakeTools({}))
    assert out == "failed"
    assert store.run.error == "llm_timeout"


def test_tool_error_fails_and_marks_step_failed(monkeypatch):
    store = _mk()
    _patch(monkeypatch, store)
    provider = FakeProvider([
        LLMResult(text="call", stop_reason="tool_use", tokens=1, tool_name="boom", tool_args={}),
    ])
    tools = FakeTools({"boom": EFFECT_READ}, behavior={"boom": "raise"})
    out = _run(store, provider, tools)
    assert out == "failed"
    assert store.run.error.startswith("tool_error")
    assert store.steps[-1].kind == "tool" and store.steps[-1].status == "failed"


def test_cancel_between_steps(monkeypatch):
    store = _mk()
    monkeypatch.setattr(agent_runner, "runs_repo", store)

    async def _cancelled(rid):
        return True

    monkeypatch.setattr(agent_runner.redis_bus, "is_run_cancelled", _cancelled)
    out = _run(store, FakeProvider([LLMResult(text="x", stop_reason="end", tokens=1)]), FakeTools({}))
    assert out == "cancelled"
    assert store.run.status == "cancelled"
    assert store.agent.status == "idle"


def test_max_steps_guard(monkeypatch):
    store = _mk()
    _patch(monkeypatch, store)
    monkeypatch.setattr(settings, "run_max_steps", 2)
    # a provider that always wants another tool call → would loop forever without the cap
    provider = FakeProvider([LLMResult(text="more", stop_reason="tool_use", tokens=1, tool_name="echo", tool_args={}) for _ in range(10)])
    out = _run(store, provider, FakeTools({"echo": EFFECT_READ}))
    assert out == "failed"
    assert store.run.error == "max_steps_exceeded"
