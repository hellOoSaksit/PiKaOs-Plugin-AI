"""LLM provider config HTTP routes (no-hardcode) — `/api/ai/llm/connections` (§6: every route
this plugin owns is namespaced under `/ai`, or the loader quarantines the whole plugin).

Admin-managed: which provider (Local/Ollama vs OpenAI vs Anthropic), model, endpoint, and key
the engine uses — set from the UI instead of `.env`. Permission split: reads require `llm.view`,
connection writes require `llm.manage`, and binding a connection to a system role requires
`llm.assign` (a role granted a write perm should also hold `llm.view` so the panel can load).
The API key is write-only (sent in, never returned — `DocumentOut`-style masking in the service).
"""
from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from ...core.db import get_db
from ...core.identity import require_perm
from .schemas import (
    LlmConnectionIn,
    LlmConnectionOut,
    LlmConnectionUpdate,
    LlmRoleOut,
    LlmRoleSet,
    LlmTestOut,
)
from . import llm_config_service as svc

router = APIRouter(prefix="/api/ai/llm/connections", tags=["llm-config"])
roles_router = APIRouter(prefix="/api/ai/llm/roles", tags=["llm-config"])


def _bad(e: svc.BadProvider) -> HTTPException:
    return HTTPException(status.HTTP_400_BAD_REQUEST, str(e))


def _test_failed(e: svc.TestFailed) -> HTTPException:
    """Save-time connection test failed → 422 with the sanitized probe result (category + short
    detail) so the form can show WHY inline. The row was not persisted."""
    return HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, detail={"test": e.result})


@router.get("", response_model=list[LlmConnectionOut])
async def list_connections(
    _: object = Depends(require_perm("llm.view")),
    db: AsyncSession = Depends(get_db),
) -> list[LlmConnectionOut]:
    return [LlmConnectionOut(**o) for o in await svc.list_out(db)]


@router.post("", response_model=LlmConnectionOut, status_code=status.HTTP_201_CREATED)
async def create_connection(
    body: LlmConnectionIn,
    _: object = Depends(require_perm("llm.manage")),
    db: AsyncSession = Depends(get_db),
) -> LlmConnectionOut:
    try:
        out = await svc.create(db, name=body.name, provider=body.provider, model=body.model,
                               base_url=body.base_url, api_key=body.api_key)
    except svc.BadProvider as e:
        raise _bad(e)
    except svc.TestFailed as e:
        raise _test_failed(e)
    return LlmConnectionOut(**out)


@router.patch("/{cid}", response_model=LlmConnectionOut)
async def update_connection(
    cid: uuid.UUID,
    body: LlmConnectionUpdate,
    _: object = Depends(require_perm("llm.manage")),
    db: AsyncSession = Depends(get_db),
) -> LlmConnectionOut:
    try:
        out = await svc.update(db, cid, name=body.name, provider=body.provider, model=body.model,
                               base_url=body.base_url, api_key=body.api_key)
    except svc.BadProvider as e:
        raise _bad(e)
    except svc.TestFailed as e:
        raise _test_failed(e)
    except svc.NotFound:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "connection not found")
    return LlmConnectionOut(**out)


@router.post("/{cid}/activate", response_model=LlmConnectionOut)
async def activate_connection(
    cid: uuid.UUID,
    _: object = Depends(require_perm("llm.manage")),
    db: AsyncSession = Depends(get_db),
) -> LlmConnectionOut:
    try:
        out = await svc.activate(db, cid)
    except svc.NotFound:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "connection not found")
    return LlmConnectionOut(**out)


@router.post("/{cid}/test", response_model=LlmTestOut)
async def test_connection(
    cid: uuid.UUID,
    _: object = Depends(require_perm("llm.manage")),
    db: AsyncSession = Depends(get_db),
) -> LlmTestOut:
    """Probe a saved connection's endpoint + stored key (no completion tokens spent). The key never
    leaves the server; the result is sanitized (a category + short message, never a raw body)."""
    try:
        out = await svc.test_connection(db, cid)
    except svc.NotFound:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "connection not found")
    return LlmTestOut(**out)


@router.delete("/{cid}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_connection(
    cid: uuid.UUID,
    _: object = Depends(require_perm("llm.manage")),
    db: AsyncSession = Depends(get_db),
) -> None:
    try:
        await svc.delete(db, cid)
    except svc.NotFound:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "connection not found")


# --- per-system role assignment (which connection a feature uses) ------------


@roles_router.get("", response_model=list[LlmRoleOut])
async def list_roles(
    _: object = Depends(require_perm("llm.view")),
    db: AsyncSession = Depends(get_db),
) -> list[LlmRoleOut]:
    return [LlmRoleOut(**o) for o in await svc.roles_out(db)]


@roles_router.put("/{role}", response_model=LlmRoleOut)
async def set_role(
    role: str,
    body: LlmRoleSet,
    _: object = Depends(require_perm("llm.assign")),
    db: AsyncSession = Depends(get_db),
) -> LlmRoleOut:
    try:
        out = await svc.set_role(db, role, body.connection_id)
    except svc.BadRole as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e))
    except svc.NotFound:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "connection not found")
    return LlmRoleOut(**out)
