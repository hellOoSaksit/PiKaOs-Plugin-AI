"""Task stream service — authorization + snapshot/backfill for the live worklog (B5).

The WS router (routers/ws.py) calls these to decide whether a socket may subscribe to a
task, and to replay state so a mid-run page open / reconnect loses nothing
(system-design §6). Pure orchestration over repositories; no FastAPI/WS types in/out.
"""
from __future__ import annotations

import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from . import tasks_repo
from . import runs_repo
from ...core.identity import ADMIN_ROLE
from .events import serialize_step


async def can_view(db: AsyncSession, user_id: str, task_id: str) -> bool:
    """May this user view this task's stream? admin · the task's creator · or a member of
    the task's department. (A dept-less task is owner/admin-only — depts are seeded in D.)

    User role + department membership are logical reads of the auth-owned tables (via tasks_repo) —
    Core no longer imports the auth User model."""
    try:
        uid, qid = uuid.UUID(user_id), uuid.UUID(task_id)
    except (ValueError, TypeError):
        return False
    role = await tasks_repo.user_role(db, uid)
    task = await tasks_repo.get_task(db, qid)
    if role is None or task is None:
        return False
    if role == ADMIN_ROLE or task.created_by == uid:
        return True
    if task.department_id is not None and await tasks_repo.user_in_department(db, uid, task.department_id):
        return True
    return False


async def snapshot(db: AsyncSession, task_id: str, *, limit: int = 200) -> dict:
    """Recent runs + worklog steps for a task — sent right after a successful subscribe."""
    qid = uuid.UUID(task_id)
    runs = await tasks_repo.run_states_for_task(db, qid)
    steps = await tasks_repo.recent_steps_for_task(db, qid, limit=limit)
    return {
        "type": "snapshot",
        "task_id": task_id,
        "runs": [{"run_id": str(rid), "status": status} for rid, status in runs],
        "steps": [serialize_step(s) for s in steps],
    }


async def backfill(db: AsyncSession, task_id: str, run_id: str, after_seq: int) -> dict:
    """Steps of one run with seq > after_seq — fills a gap the client detected via (run_id, seq).

    The caller has already passed `can_view` for `task_id`; here we additionally confirm the
    requested run actually belongs to that task, so a crafted run_id can't read another task.
    """
    empty = {"type": "backfill", "run_id": run_id, "after_seq": after_seq, "steps": []}
    try:
        rid, qid = uuid.UUID(run_id), uuid.UUID(task_id)
    except (ValueError, TypeError):
        return empty
    run = await runs_repo.get_run(db, rid)
    if run is None or run.task_id != qid:
        return empty
    steps = await tasks_repo.steps_after(db, rid, after_seq)
    return {**empty, "steps": [serialize_step(s) for s in steps]}
