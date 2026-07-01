"""Engine → browser event stream (B5).

The runner publishes one event per step (and per run-status change) to Redis channel
`task:<id>`; the WS relay (routers/ws.py, A2) forwards them to every socket subscribed to
that task → a live worklog timeline. Every step event carries `(run_id, seq)` so the client
can detect a gap and ask for a `backfill` (system-design §6).

Publishing is **best-effort**: a Redis outage must never fail the run (the worklog is also
durably in `run_steps`, so a reconnect's snapshot recovers it). `serialize_step` is shared
with the snapshot/backfill path in task_service so the wire shape is identical everywhere.
"""
from __future__ import annotations

import json
import logging

from redis.exceptions import RedisError

from ...core.redis_client import redis

log = logging.getLogger("pikaos.events")

_TASK_CHANNEL = "task:{}"
# Keep events small (system-design §6: ≤ ~32KB; large tool output → MinIO object_key later).
_CONTENT_CAP_BYTES = 16 * 1024


def _cap_content(content: dict | None) -> dict | None:
    if content is None:
        return None
    try:
        size = len(json.dumps(content, default=str))
    except (TypeError, ValueError):
        return {"truncated": True, "reason": "unserializable"}
    if size > _CONTENT_CAP_BYTES:
        return {"truncated": True, "bytes": size}
    return content


def serialize_step(step) -> dict:
    """Wire shape of one run_step — used by both live events and snapshot/backfill."""
    return {
        "id": str(step.id),
        "run_id": str(step.run_id),
        "seq": step.seq,
        "kind": step.kind,
        "status": step.status,
        "role": step.role,
        "tokens": step.tokens,
        "content": _cap_content(step.content),
    }


async def _publish(task_id: str | None, event: dict) -> None:
    if not task_id:
        return  # a run not bound to a task streams nowhere — nothing to do
    try:
        await redis.publish(_TASK_CHANNEL.format(task_id), json.dumps(event, default=str))
    except RedisError as exc:
        log.warning("redis down — dropped task event (durable in run_steps): %s", exc)


async def publish_step(task_id: str | None, step) -> None:
    await _publish(task_id, {"type": "step", "task_id": task_id, **serialize_step(step)})


async def publish_run(task_id: str | None, run_id, status: str, **extra) -> None:
    await _publish(task_id, {"type": "run", "task_id": task_id, "run_id": str(run_id), "status": status, **extra})
