"""Tests for the B4 engine stubs — scripted LLM provider + effect-classed tools.

* StubLLMProvider is pure (no DB) → driven with asyncio.run.
* The stub tools' substance is the dedup SQL (`stub_tool_writes`): record = one row per
  call (side_effect, at-most-once) vs upsert = ON CONFLICT DO NOTHING (idempotent_write,
  replay-safe). Those hit the real DB via a fresh engine created inside asyncio.run, which
  sidesteps the module-level-engine event-loop issue (see tests/conftest note in test_compare).
  The full kill-worker-mid-step run on Postgres is B6.

    docker compose exec backend pytest tests/test_engine_stubs.py
"""
from __future__ import annotations

import asyncio
import json
import uuid

from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import settings
from app.plugins.ai.models import Run, StubToolWrite
from app.plugins.ai import stub_tools_repo as stub_repo
from app.plugins.ai.agent_runner import EFFECT_IDEMPOTENT_WRITE, EFFECT_READ, EFFECT_SIDE_EFFECT
from app.plugins.ai.engine_stubs import SCRIPT_SENTINEL, StubLLMProvider, StubToolRegistry


def _script_msg(specs: list[dict]) -> dict:
    return {"role": "user", "content": SCRIPT_SENTINEL + json.dumps(specs)}


def _complete(messages):
    return asyncio.run(StubLLMProvider().complete(model="stub", messages=messages, tools=[]))


# --- StubLLMProvider (pure) -------------------------------------------------


def test_provider_scripts_a_tool_then_finishes():
    specs = [{"text": "calling", "tool": "record", "args": {"a": 1}, "tokens": 7}, {"text": "wrap", "tokens": 2}]
    msg = _script_msg(specs)

    r0 = _complete([msg])  # turn 0 (no assistant yet)
    assert r0.stop_reason == "tool_use" and r0.tool_name == "record" and r0.tool_args == {"a": 1} and r0.tokens == 7

    # turn 1: one assistant message has happened
    r1 = _complete([msg, {"role": "assistant", "content": "calling"}, {"role": "tool", "content": {}}])
    assert r1.stop_reason == "end" and r1.text == "wrap" and r1.tokens == 2


def test_provider_exhausted_script_finalizes():
    msg = _script_msg([{"text": "only one"}])
    r = _complete([msg, {"role": "assistant", "content": "only one"}])  # turn 1 > len 1
    assert r.stop_reason == "end"


def test_provider_default_echo_without_script():
    r = _complete([{"role": "user", "content": "hello world"}])
    assert r.stop_reason == "end" and "hello world" in r.text and r.tokens >= 1


def test_provider_default_token_estimate():
    r = _complete([_script_msg([{"text": "abcdefgh"}])])  # 8 chars, no explicit tokens
    assert r.tokens == 2  # len // 4


# --- StubToolRegistry effect mapping ---------------------------------------


def test_registry_effect_classes():
    reg = StubToolRegistry()
    assert reg.effect_of("echo") == EFFECT_READ
    assert reg.effect_of("upsert") == EFFECT_IDEMPOTENT_WRITE
    assert reg.effect_of("record") == EFFECT_SIDE_EFFECT
    assert reg.effect_of("anything_else") == EFFECT_SIDE_EFFECT  # safe default
    assert {s["name"] for s in reg.schemas()} == {"echo", "upsert", "record"}


def test_registry_echo_is_db_free():
    out = asyncio.run(StubToolRegistry().call("echo", {"x": 1}, idempotency_key="k"))
    assert out == {"echo": {"x": 1}}


# --- stub tool sink dedup (real DB) ----------------------------------------


def _run_db(coro_fn, rid):
    """Seed a real runs row (stub_tool_writes.run_id FK), run the scenario, then clean up —
    each in its own session so an aborted scenario transaction can't break teardown."""
    async def main():
        eng = create_async_engine(settings.database_url)
        Session = async_sessionmaker(eng, expire_on_commit=False, class_=AsyncSession)
        try:
            async with Session() as setup:
                setup.add(Run(id=rid))
                await setup.commit()
            async with Session() as db:
                return await coro_fn(db)
        finally:
            async with Session() as cleanup:
                await cleanup.execute(delete(StubToolWrite).where(StubToolWrite.run_id == rid))
                await cleanup.execute(delete(Run).where(Run.id == rid))
                await cleanup.commit()
            await eng.dispose()

    return asyncio.run(main())


def test_record_writes_one_row_per_call():
    rid = uuid.uuid4()

    async def scen(db):
        await stub_repo.record_write(db, rid, "record", f"{rid}:0", {"a": 1})
        await stub_repo.record_write(db, rid, "record", f"{rid}:1", {"a": 2})
        return await stub_repo.count_writes(db, rid)

    assert _run_db(scen, rid) == 2


def test_upsert_dedups_same_key():
    rid = uuid.uuid4()

    async def scen(db):
        key = f"{rid}:0"
        first = await stub_repo.upsert_write(db, rid, "upsert", key, {"v": 1})
        second = await stub_repo.upsert_write(db, rid, "upsert", key, {"v": 2})  # same key → no-op
        return first, second, await stub_repo.count_writes(db, rid)

    first, second, n = _run_db(scen, rid)
    assert first is True and second is False and n == 1
