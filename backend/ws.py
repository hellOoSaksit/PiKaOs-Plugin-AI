"""WebSocket endpoint — first-message auth + per-channel relay over Redis pub/sub.

Security (A2 / risk-mitigation §3): the access token is NEVER in the URL — that leaks into
proxy/access logs. The client connects to /ws, then its FIRST frame must be
``{"type":"auth","token":"<access JWT>"}`` within AUTH_TIMEOUT seconds, else the socket is
closed 4401. After auth the socket subscribes to its own user channel (``pikaos:user:<id>``);
the old scaffold relayed one global channel to every logged-in user (cross-user leak).

Task streaming (``{"type":"subscribe","task_id":...}``) is live (B5): the socket is
authorized via ``task_service.can_view`` (owner / department member / admin), subscribed to
Redis ``task:<id>`` where the runner publishes one event per step, and sent a snapshot of
recent runs+steps so a mid-run page open loses nothing. ``{"type":"backfill","run_id",
"after_seq"}`` fills a gap the client detects via each event's ``(run_id, seq)``
(system-design §6 · risk-mitigation §3 ค–ง).
"""
from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ...core import redis_client
from ...core.db import SessionLocal
from ...core.identity import provider_for
from . import task_service

router = APIRouter()

AUTH_TIMEOUT = 5.0                       # seconds to send the first {"type":"auth"} frame
_USER_CHANNEL = "pikaos:user:{}"
_TASK_CHANNEL = "task:{}"


async def _authenticate(websocket: WebSocket) -> str | None:
    """Wait for the first-message auth frame; return the user_id, or None to reject."""
    try:
        raw = await asyncio.wait_for(websocket.receive_text(), timeout=AUTH_TIMEOUT)
    except (asyncio.TimeoutError, WebSocketDisconnect):
        return None
    try:
        msg = json.loads(raw)
    except (ValueError, TypeError):
        return None
    if msg.get("type") != "auth":
        return None
    # Delegate token validation to the identity provider (auth plugin) — the kernel router no longer
    # decodes JWTs itself. The provider checks signature, type, jti-denylist and active status, returning
    # a user or None; unbound provider (auth off) → BootstrapProvider denies → None.
    user = await provider_for(websocket.app).authenticate(msg.get("token", ""))
    return str(user.id) if user is not None else None


async def _can_view_task(user_id: str, task_id: str) -> bool:
    async with SessionLocal() as db:
        return await task_service.can_view(db, user_id, task_id)


@router.websocket("/ws")
async def ws_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    user_id = await _authenticate(websocket)
    if user_id is None:
        await websocket.close(code=4401)  # unauthorized
        return

    pubsub = redis_client.redis.pubsub()
    channels = {_USER_CHANNEL.format(user_id)}
    await pubsub.subscribe(*channels)

    async def relay() -> None:
        async for message in pubsub.listen():
            if message.get("type") == "message":
                await websocket.send_text(message["data"])

    relay_task = asyncio.create_task(relay())
    try:
        await websocket.send_json({"type": "ready", "user": user_id})
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except (ValueError, TypeError):
                continue
            kind = msg.get("type")
            if kind == "subscribe":
                task_id = str(msg.get("task_id", ""))
                if task_id and await _can_view_task(user_id, task_id):
                    ch = _TASK_CHANNEL.format(task_id)
                    channels.add(ch)
                    # Subscribe BEFORE the snapshot so no event published in the gap is missed;
                    # the client dedups by (run_id, seq) if the snapshot and a live event overlap.
                    await pubsub.subscribe(ch)
                    await websocket.send_json({"type": "subscribed", "task_id": task_id})
                    async with SessionLocal() as db:
                        await websocket.send_json(await task_service.snapshot(db, task_id))
                else:
                    await websocket.send_json({"type": "error", "reason": "forbidden", "task_id": task_id})
            elif kind == "unsubscribe":
                task_id = str(msg.get("task_id", ""))
                ch = _TASK_CHANNEL.format(task_id)
                if ch in channels:
                    channels.discard(ch)
                    await pubsub.unsubscribe(ch)
                    await websocket.send_json({"type": "unsubscribed", "task_id": task_id})
            elif kind == "backfill":
                # Gap recovery: client asks for steps of a run it's authorized to see, after a seq.
                run_id, after_seq = str(msg.get("run_id", "")), int(msg.get("after_seq", -1))
                task_id = str(msg.get("task_id", ""))
                if run_id and task_id and _TASK_CHANNEL.format(task_id) in channels:
                    async with SessionLocal() as db:
                        await websocket.send_json(await task_service.backfill(db, task_id, run_id, after_seq))
            # any other frame is ignored — no global echo (that was the cross-user leak)
    except WebSocketDisconnect:
        pass
    finally:
        relay_task.cancel()
        if channels:
            await pubsub.unsubscribe(*channels)
        await pubsub.aclose()
