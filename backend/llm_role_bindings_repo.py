"""SQL for `llm_role_bindings` — per-system LLM assignment (no-hardcode, server-scoped).

One row per system role (engine/search/summarize) → the `llm_connections` row it uses.
A missing row means "fall back to the active connection" (resolved in the service layer).
"""
from __future__ import annotations

import uuid

from sqlalchemy import delete as sql_delete
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .models import LlmConnection, LlmRoleBinding


async def list_bindings(db: AsyncSession) -> dict[str, uuid.UUID]:
    """role -> connection_id for every bound role."""
    rows = (await db.execute(select(LlmRoleBinding))).scalars().all()
    return {r.role: r.connection_id for r in rows}


async def get_connection_for_role(db: AsyncSession, role: str) -> LlmConnection | None:
    """The connection a role is bound to (or None if unbound / the connection is gone)."""
    stmt = (
        select(LlmConnection)
        .join(LlmRoleBinding, LlmRoleBinding.connection_id == LlmConnection.id)
        .where(LlmRoleBinding.role == role)
        .limit(1)
    )
    return (await db.execute(stmt)).scalars().first()


async def set_binding(db: AsyncSession, role: str, connection_id: uuid.UUID) -> None:
    """Upsert a role -> connection binding."""
    row = await db.get(LlmRoleBinding, role)
    if row is None:
        db.add(LlmRoleBinding(role=role, connection_id=connection_id))
    else:
        row.connection_id = connection_id
    await db.commit()


async def clear_binding(db: AsyncSession, role: str) -> None:
    """Remove a role binding (the role then falls back to the active connection)."""
    await db.execute(sql_delete(LlmRoleBinding).where(LlmRoleBinding.role == role))
    await db.commit()
