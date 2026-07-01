"""Redis for the ai engine — the realtime pub/sub client + run-cancellation flags.

These left the kernel with the Redis extraction: the task/user event stream (events.py publishes, ws.py
subscribes) and the cooperative run-cancel flag (agent_runner checks it between steps) are AI concerns,
not kernel infra. The aioredis client is resolved from the `redis.Connection` DI contract (bound by the
redis tool) at this plugin's register() and stashed in `_redis`. Access it via `client()` at call time
(never `from . import _redis`) so callers see the bound client, and tolerate `None` (redis tool disabled /
not yet bound) the same way they already tolerate a Redis outage.
"""
from __future__ import annotations

import logging

from redis.exceptions import RedisError

log = logging.getLogger("pikaos.ai.redis")

_redis = None  # the aioredis client, bound from redis.Connection at register()


def bind(conn) -> None:
    """Wire the aioredis client resolved from `redis.Connection` (called by the plugin's register())."""
    global _redis
    _redis = conn


def client():
    """The bound aioredis client, or None when the redis tool is disabled/unbound (callers must guard)."""
    return _redis


# --- run cancellation ---
# The runner checks this between steps; setting it asks an in-flight run to stop at the next boundary
# (mid-step cancellation rides on the per-step timeouts). Short TTL so a stale flag can't cancel a future
# run that happens to reuse the id.
_CANCEL = "run:{}:cancel"
_CANCEL_TTL = 60 * 60  # 1h — longer than any run's wall-clock ceiling


async def request_run_cancel(run_id: str) -> None:
    if _redis is None:
        return
    try:
        await _redis.set(_CANCEL.format(run_id), "1", ex=_CANCEL_TTL)
    except RedisError as exc:
        log.warning("redis down — could not flag run cancel: %s", exc)


async def is_run_cancelled(run_id: str) -> bool:
    # Fail-open (treat as not-cancelled) on a Redis outage: a run we can't check should finish on its own
    # bounds (max_steps / wall-clock) rather than be force-killed.
    if _redis is None:
        return False
    try:
        return await _redis.exists(_CANCEL.format(run_id)) == 1
    except RedisError as exc:
        log.warning("redis down — skipping cancel check (fail-open): %s", exc)
        return False


async def clear_run_cancel(run_id: str) -> None:
    if _redis is None:
        return
    try:
        await _redis.delete(_CANCEL.format(run_id))
    except RedisError as exc:
        log.warning("redis down — could not clear run cancel flag: %s", exc)
