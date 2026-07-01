"""SQL for `llm_connections` — runtime-configurable LLM providers (server-scoped, no-hardcode).

One row may be `is_active` (enforced by a partial unique index, migration 0003); the worker
reads the active one to build its provider. API keys are stored encrypted by the service layer.
"""
from __future__ import annotations

import uuid

from sqlalchemy import delete as sql_delete
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from .models import LlmConnection


async def list_connections(db: AsyncSession) -> list[LlmConnection]:
    stmt = select(LlmConnection).order_by(LlmConnection.created_at.desc())
    return list((await db.execute(stmt)).scalars().all())


async def get_connection(db: AsyncSession, cid: uuid.UUID) -> LlmConnection | None:
    return await db.get(LlmConnection, cid)


async def get_active(db: AsyncSession) -> LlmConnection | None:
    stmt = select(LlmConnection).where(LlmConnection.is_active.is_(True)).limit(1)
    return (await db.execute(stmt)).scalars().first()


async def insert_connection(
    db: AsyncSession, *, name: str, provider: str, model: str,
    base_url: str | None, api_key_enc: str | None,
) -> LlmConnection:
    row = LlmConnection(id=uuid.uuid4(), name=name, provider=provider, model=model,
                        base_url=base_url, api_key_enc=api_key_enc, is_active=False)
    db.add(row)
    await db.commit()
    await db.refresh(row)
    return row


async def update_connection(db: AsyncSession, cid: uuid.UUID, **fields) -> LlmConnection | None:
    row = await db.get(LlmConnection, cid)
    if row is None:
        return None
    for k, v in fields.items():
        if v is not None:  # None = "leave unchanged" (e.g. api key not re-entered)
            setattr(row, k, v)
    await db.commit()
    await db.refresh(row)
    return row


async def delete_connection(db: AsyncSession, cid: uuid.UUID) -> bool:
    res = await db.execute(sql_delete(LlmConnection).where(LlmConnection.id == cid))
    await db.commit()
    return res.rowcount > 0


async def set_active(db: AsyncSession, cid: uuid.UUID) -> LlmConnection | None:
    """Make one connection active (and only one — clear the rest first)."""
    row = await db.get(LlmConnection, cid)
    if row is None:
        return None
    await db.execute(update(LlmConnection).values(is_active=False))
    row.is_active = True
    await db.commit()
    await db.refresh(row)
    return row
