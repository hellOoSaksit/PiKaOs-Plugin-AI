"""SQL for the B4 stub tool sink (`stub_tool_writes`). Engine self-test fixture only."""
from __future__ import annotations

import uuid

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from .models import StubToolWrite


async def record_write(db: AsyncSession, run_id: uuid.UUID | None, tool: str, key: str, payload: dict) -> bool:
    """side_effect tool: a plain INSERT — one row per call. (The runner guarantees a
    side_effect tool is dispatched at most once, so this never double-writes on resume.)"""
    db.add(StubToolWrite(run_id=run_id, tool=tool, idempotency_key=key, payload=payload))
    await db.commit()
    return True


async def upsert_write(db: AsyncSession, run_id: uuid.UUID | None, tool: str, key: str, payload: dict) -> bool:
    """idempotent_write tool: INSERT … ON CONFLICT(idempotency_key) DO NOTHING. A replay-safe
    resume re-runs with the same key and gets deduped → still one row. Returns True if inserted."""
    stmt = (
        pg_insert(StubToolWrite)
        .values(id=uuid.uuid4(), run_id=run_id, tool=tool, idempotency_key=key, payload=payload)
        .on_conflict_do_nothing(index_elements=["idempotency_key"])
    )
    res = await db.execute(stmt)
    await db.commit()
    return (res.rowcount or 0) > 0


async def count_writes(db: AsyncSession, run_id: uuid.UUID) -> int:
    stmt = select(func.count()).select_from(StubToolWrite).where(StubToolWrite.run_id == run_id)
    return int((await db.execute(stmt)).scalar_one())
