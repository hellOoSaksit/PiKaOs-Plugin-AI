"""Tests for the live worklog stream (B5) — event serialization, best-effort publish,
task-view authz, and the backfill cross-task guard.

* serialize/cap + best-effort publish: pure / monkeypatched (no DB).
* authz + backfill: real schema via a fresh engine inside asyncio.run (the local-engine
  technique from test_engine_stubs, dodging the module-engine loop issue). The full
  subscribe→snapshot→live-event path over a real socket is exercised by the B6 harness.

    docker compose exec backend pytest tests/test_task_stream.py
"""
from __future__ import annotations

import asyncio
import uuid
from types import SimpleNamespace

from redis.exceptions import RedisError
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import settings
from app.plugins.ai.models import Task
from app.plugins.auth.models import Department, User, UserDepartment
from app.plugins.ai import events, task_service


# --- serialize / cap (pure) -------------------------------------------------


def test_serialize_step_shape():
    step = SimpleNamespace(id=uuid.uuid4(), run_id=uuid.uuid4(), seq=3, kind="llm",
                           status="done", role="assistant", tokens=5, content={"text": "hi"})
    out = events.serialize_step(step)
    assert out == {
        "id": str(step.id), "run_id": str(step.run_id), "seq": 3, "kind": "llm",
        "status": "done", "role": "assistant", "tokens": 5, "content": {"text": "hi"},
    }


def test_cap_content_truncates_large_payloads():
    big = {"blob": "x" * 20000}
    capped = events._cap_content(big)
    assert capped["truncated"] is True and capped["bytes"] > 16 * 1024
    assert events._cap_content({"small": 1}) == {"small": 1}
    assert events._cap_content(None) is None


def test_publish_is_best_effort_on_redis_outage(monkeypatch):
    class _DownRedis:
        async def publish(self, *a, **k):
            raise RedisError("down")

    # redis moved to the redis Tool: events resolves the client via redis_bus.client() (bound in register())
    monkeypatch.setattr(events.redis_bus, "client", lambda: _DownRedis())
    # must not raise — the worklog is durable in run_steps; a reconnect's snapshot recovers it
    asyncio.run(events.publish_run("q1", uuid.uuid4(), "done"))
    asyncio.run(events.publish_step("q1", SimpleNamespace(
        id=uuid.uuid4(), run_id=uuid.uuid4(), seq=0, kind="llm", status="done", role=None, tokens=0, content=None)))


def test_publish_skips_when_no_task(monkeypatch):
    published = []

    class _Rec:
        async def publish(self, ch, data):
            published.append(ch)

    monkeypatch.setattr(events.redis_bus, "client", lambda: _Rec())
    asyncio.run(events.publish_run(None, uuid.uuid4(), "done"))
    assert published == []  # a run with no task streams nowhere


# --- authz + backfill (real DB) ---------------------------------------------


def _run_db(coro_fn):
    async def main():
        eng = create_async_engine(settings.database_url)
        Session = async_sessionmaker(eng, expire_on_commit=False, class_=AsyncSession)
        try:
            async with Session() as db:
                return await coro_fn(db, Session)
        finally:
            await eng.dispose()

    return asyncio.run(main())


def _user(role="member"):
    u = uuid.uuid4()
    return User(id=u, username=f"u_{u.hex[:8]}", email=f"{u.hex[:8]}@t.local",
                display="t", role=role, password_hash="x")


def test_can_view_owner_admin_and_denies_outsider():
    owner, outsider, admin = _user(), _user(), _user(role="admin")
    qid = uuid.uuid4()

    async def scen(db, Session):
        db.add_all([owner, outsider, admin])
        await db.flush()  # parents before the FK child (no ORM relationship to auto-order)
        db.add(Task(id=qid, title="t", created_by=owner.id))
        await db.commit()
        try:
            return (
                await task_service.can_view(db, str(owner.id), str(qid)),     # owner → True
                await task_service.can_view(db, str(admin.id), str(qid)),     # admin → True
                await task_service.can_view(db, str(outsider.id), str(qid)),  # outsider → False
                await task_service.can_view(db, str(owner.id), str(uuid.uuid4())),  # missing task → False
            )
        finally:
            await db.execute(delete(Task).where(Task.id == qid))
            await db.execute(delete(User).where(User.id.in_([owner.id, outsider.id, admin.id])))
            await db.commit()

    is_owner, is_admin, is_outsider, missing = _run_db(scen)
    assert is_owner is True and is_admin is True and is_outsider is False and missing is False


def test_can_view_department_member():
    member, other = _user(), _user()
    dept = uuid.uuid4()
    qid = uuid.uuid4()

    async def scen(db, Session):
        db.add_all([member, other, Department(id=dept, name_en="Eng")])
        await db.flush()  # users + department before their FK children
        db.add_all([
            UserDepartment(user_id=member.id, department_id=dept),
            Task(id=qid, title="t", created_by=other.id, department_id=dept),  # owned by other, in member's dept
        ])
        await db.commit()
        try:
            return (
                await task_service.can_view(db, str(member.id), str(qid)),  # dept member → True
                await task_service.can_view(db, str(other.id), str(qid)),   # owner → True
            )
        finally:
            await db.execute(delete(Task).where(Task.id == qid))
            await db.execute(delete(UserDepartment).where(UserDepartment.department_id == dept))
            await db.execute(delete(Department).where(Department.id == dept))
            await db.execute(delete(User).where(User.id.in_([member.id, other.id])))
            await db.commit()

    member_ok, owner_ok = _run_db(scen)
    assert member_ok is True and owner_ok is True


def test_backfill_rejects_foreign_run():
    owner = _user()
    qa, qb = uuid.uuid4(), uuid.uuid4()
    from app.plugins.ai.models import Run

    async def scen(db, Session):
        db.add(owner)
        await db.flush()  # owner before the tasks that reference it
        db.add_all([Task(id=qa, title="a", created_by=owner.id), Task(id=qb, title="b", created_by=owner.id)])
        await db.flush()  # tasks before the run that references one
        run_a = Run(id=uuid.uuid4(), task_id=qa, status="done")
        db.add(run_a)
        await db.commit()
        try:
            # asking for run_a's steps but claiming task B → guarded to empty
            foreign = await task_service.backfill(db, str(qb), str(run_a.id), -1)
            # same run under its real task → allowed (no steps here, but not rejected)
            native = await task_service.backfill(db, str(qa), str(run_a.id), -1)
            return foreign["steps"], native
        finally:
            await db.execute(delete(Run).where(Run.id == run_a.id))
            await db.execute(delete(Task).where(Task.id.in_([qa, qb])))
            await db.execute(delete(User).where(User.id == owner.id))
            await db.commit()

    foreign_steps, native = _run_db(scen)
    assert foreign_steps == []
    assert native["type"] == "backfill" and native["run_id"]
