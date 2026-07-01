"""Run / run_step queries — all SQL for the agent engine lives here (layering §2.1).

The agent_runner service orchestrates these; it never writes SQL itself. Token quota is
reserved with a single atomic UPDATE (no read-then-add race across concurrent runs) — see
`reserve_quota`.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import func, select, text, update
from sqlalchemy.ext.asyncio import AsyncSession

from .models import Agent, Run, RunStep


async def get_run(db: AsyncSession, run_id: uuid.UUID) -> Run | None:
    return await db.get(Run, run_id)


async def get_agent(db: AsyncSession, agent_id: uuid.UUID) -> Agent | None:
    return await db.get(Agent, agent_id)


async def list_steps(db: AsyncSession, run_id: uuid.UUID) -> list[RunStep]:
    """Worklog steps in deterministic order — used to reconstruct a run on resume."""
    stmt = select(RunStep).where(RunStep.run_id == run_id).order_by(RunStep.seq)
    return list((await db.execute(stmt)).scalars().all())


async def set_run_status(
    db: AsyncSession,
    run_id: uuid.UUID,
    status: str,
    *,
    started: bool = False,
    ended: bool = False,
    error: str | None = None,
) -> None:
    values: dict = {"status": status}
    now = datetime.now(timezone.utc)
    if started:
        values["started_at"] = now
    if ended:
        values["ended_at"] = now
    if error is not None:
        values["error"] = error
    await db.execute(update(Run).where(Run.id == run_id).values(**values))
    await db.commit()


async def set_agent_status(db: AsyncSession, agent_id: uuid.UUID | None, status: str) -> None:
    """Agent status is runner-set only (product rule). No-op when the run has no agent."""
    if agent_id is None:
        return
    await db.execute(update(Agent).where(Agent.id == agent_id).values(status=status))
    await db.commit()


async def insert_step(
    db: AsyncSession,
    run_id: uuid.UUID,
    seq: int,
    kind: str,
    *,
    status: str = "done",
    idempotency_key: str | None = None,
    role: str | None = None,
    content: dict | None = None,
    tokens: int = 0,
) -> RunStep:
    """Append one worklog step. UNIQUE(run_id, seq) guards against double-append on resume."""
    step = RunStep(
        run_id=run_id,
        seq=seq,
        kind=kind,
        status=status,
        idempotency_key=idempotency_key,
        role=role,
        content=content,
        tokens=tokens,
    )
    db.add(step)
    await db.commit()
    await db.refresh(step)
    return step


async def complete_step(
    db: AsyncSession,
    step_id: uuid.UUID,
    *,
    status: str = "done",
    content: dict | None = None,
    tokens: int = 0,
) -> None:
    """Phase-2 of a two-phase tool step: flip pending → done/failed with the result."""
    values: dict = {"status": status, "tokens": tokens}
    if content is not None:
        values["content"] = content
    await db.execute(update(RunStep).where(RunStep.id == step_id).values(**values))
    await db.commit()


async def add_run_tokens(db: AsyncSession, run_id: uuid.UUID, tokens: int) -> None:
    """Roll a step's tokens up to the run total (keeps runs.tokens_used == Σ its step tokens)."""
    if tokens <= 0:
        return
    await db.execute(update(Run).where(Run.id == run_id).values(tokens_used=Run.tokens_used + tokens))
    await db.commit()


async def reserve_quota(db: AsyncSession, user_id: uuid.UUID | None, tokens: int) -> bool:
    """Atomically reserve `tokens` against the owner's quota. Returns False if it would
    exceed (caller fails the run `quota_exceeded`).

    A single conditional UPDATE — no read-then-add race when several runs of the same user
    reserve concurrently. NULL quota = unlimited; a null owner / non-positive tokens reserve
    nothing. `used` therefore stays equal to the sum of persisted run_steps.tokens.
    """
    if user_id is None or tokens <= 0:
        return True
    stmt = text(
        "UPDATE users SET used = used + :n "
        "WHERE id = :uid AND (quota IS NULL OR used + :n <= quota) "
        "RETURNING used"
    )
    row = (await db.execute(stmt, {"n": tokens, "uid": str(user_id)})).first()
    await db.commit()
    return row is not None


async def total_step_tokens(db: AsyncSession, run_id: uuid.UUID) -> int:
    """Sum of a run's persisted step tokens — used in tests to assert quota == Σ steps."""
    stmt = select(func.coalesce(func.sum(RunStep.tokens), 0)).where(RunStep.run_id == run_id)
    return int((await db.execute(stmt)).scalar_one())
