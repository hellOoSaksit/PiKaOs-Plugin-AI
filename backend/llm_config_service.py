"""Runtime LLM provider config (no-hardcode, server-scoped).

Admins set the engine's provider/model/endpoint/key from the UI (จัดการเครื่องมือ) instead of
`.env`; this layer:
  - **encrypts** the API key at rest (`crypto`) and **masks** it on read (never returns plaintext),
  - resolves the *active* connection into a live `LLMProvider` for the worker, cached for a few
    seconds so an admin's change takes effect **without a worker restart**,
  - falls back to the `.env`-configured provider (settings) when no connection is active.

No SQL here (repositories/llm_connections) and no FastAPI types (routers/llm_config) — §2.1.
"""
from __future__ import annotations

import time
import uuid

from ...core import crypto
from ...core.config import settings
from . import db_ref
from . import llm_connections_repo as repo
from . import llm_role_bindings_repo as role_repo
from .engine_stubs import StubLLMProvider
from .llm_anthropic import AnthropicProvider
from .llm_ollama import OllamaProvider
from .llm_openai import OpenAIProvider

PROVIDERS = ("ollama", "openai", "anthropic")

# System roles that consume an LLM. The catalog is the *source of truth* the UI reads (no-hardcode:
# the frontend renders whatever roles the backend exposes, labelled via i18n `llmcfg.role.<key>`).
# A role with no binding falls back to the active connection — so "engine" works out of the box.
ROLES = ("engine", "search", "summarize", "answer")


class NotFound(Exception):
    """No connection with that id."""


class BadProvider(Exception):
    """Provider isn't one of PROVIDERS."""


class BadRole(Exception):
    """Role isn't one of ROLES."""


def _validate_provider(provider: str | None) -> None:
    if provider is not None and provider not in PROVIDERS:
        raise BadProvider(f"provider must be one of {PROVIDERS}")


def to_out(row) -> dict:
    """Public shape — the api key is masked to a boolean (never returned)."""
    return {
        "id": row.id, "name": row.name, "provider": row.provider, "model": row.model,
        "base_url": row.base_url, "is_active": row.is_active, "created_at": row.created_at,
        "api_key_set": bool(row.api_key_enc),
    }


async def list_out(db) -> list[dict]:
    return [to_out(r) for r in await repo.list_connections(db)]


async def create(db, *, name: str, provider: str, model: str,
                 base_url: str | None, api_key: str | None) -> dict:
    _validate_provider(provider)
    enc = crypto.encrypt(api_key) if api_key else None
    row = await repo.insert_connection(db, name=name, provider=provider, model=model or "",
                                       base_url=base_url or None, api_key_enc=enc)
    _invalidate()
    return to_out(row)


async def update(db, cid: uuid.UUID, *, name=None, provider=None, model=None,
                 base_url=None, api_key=None) -> dict:
    _validate_provider(provider)
    # only re-encrypt when a new key is actually supplied; "" / None leaves it unchanged
    enc = crypto.encrypt(api_key) if api_key else None
    row = await repo.update_connection(db, cid, name=name, provider=provider, model=model,
                                       base_url=base_url, api_key_enc=enc)
    if row is None:
        raise NotFound
    _invalidate()
    return to_out(row)


async def delete(db, cid: uuid.UUID) -> None:
    if not await repo.delete_connection(db, cid):
        raise NotFound
    _invalidate()


async def activate(db, cid: uuid.UUID) -> dict:
    row = await repo.set_active(db, cid)
    if row is None:
        raise NotFound
    _invalidate()
    return to_out(row)


# --- per-system role assignment (which connection a feature uses) ------------


async def roles_out(db) -> list[dict]:
    """One entry per known role, with its current binding (connection unbound → null)."""
    conns = {r.id: r for r in await repo.list_connections(db)}
    bound = await role_repo.list_bindings(db)
    out = []
    for role in ROLES:
        cid = bound.get(role)
        conn = conns.get(cid) if cid else None
        out.append({
            "role": role,
            "connection_id": conn.id if conn else None,        # drop dangling ids (deleted conn)
            "connection_name": conn.name if conn else None,
        })
    return out


async def set_role(db, role: str, connection_id: uuid.UUID | None) -> dict:
    """Bind a role to a connection, or clear it (connection_id=None → fall back to active)."""
    if role not in ROLES:
        raise BadRole(f"role must be one of {ROLES}")
    conn = None
    if connection_id is None:
        await role_repo.clear_binding(db, role)
    else:
        conn = await repo.get_connection(db, connection_id)
        if conn is None:
            raise NotFound
        await role_repo.set_binding(db, role, connection_id)
    _invalidate()
    return {"role": role, "connection_id": conn.id if conn else None,
            "connection_name": conn.name if conn else None}


# --- resolving the active connection into a live provider (worker side) ------


def build_provider(row) -> object:
    """Construct an LLMProvider from a connection row (decrypting its key)."""
    key = crypto.decrypt(row.api_key_enc or "")
    if row.provider == "ollama":
        return OllamaProvider(base_url=row.base_url or None, model=row.model or None)
    if row.provider == "openai":
        return OpenAIProvider(api_key=key, base_url=row.base_url or None, model=row.model or None)
    if row.provider == "anthropic":
        return AnthropicProvider(api_key=key, base_url=row.base_url or None, model=row.model or None)
    return StubLLMProvider()


def provider_from_settings() -> object:
    """Fallback when no connection is active — the .env-configured provider (C1)."""
    p = settings.llm_provider
    if p == "ollama":
        return OllamaProvider()
    if p == "openai":
        return OpenAIProvider()
    if p == "anthropic":
        return AnthropicProvider()
    return StubLLMProvider()


# resolved providers cached per role (key "" = the bare active connection), each ~llm_config_cache_s
_cache: dict = {}


def _invalidate() -> None:
    _cache.clear()


async def _resolve(role: str | None) -> object:
    """Resolve a role to a live provider: role binding → active connection → .env fallback."""
    async with db_ref.new_session() as db:
        row = await role_repo.get_connection_for_role(db, role) if role else None
        if row is None:
            row = await repo.get_active(db)
    return build_provider(row) if row is not None else provider_from_settings()


async def _cached(role: str | None) -> object:
    key = role or ""
    now = time.monotonic()
    hit = _cache.get(key)
    if hit is not None and (now - hit[0]) < settings.llm_config_cache_s:
        return hit[1]
    provider = await _resolve(role)
    _cache[key] = (now, provider)
    return provider


async def active_provider() -> object:
    """The provider for the active connection (or the .env fallback), cached ~llm_config_cache_s."""
    return await _cached(None)


async def provider_for_role(role: str) -> object:
    """The provider a system role uses — its binding, else the active connection, else .env."""
    return await _cached(role)


class ConfiguredLLMProvider:
    """LLMProvider that resolves the DB connection per call (cached) → admin edits in the UI take
    effect without a worker restart. Bound to a system role (default "engine"): the role's binding
    wins, otherwise the active connection, otherwise the .env provider. Injected into the runner."""

    def __init__(self, role: str = "engine"):
        self.role = role

    async def complete(self, *, model, messages, tools):
        provider = await provider_for_role(self.role)
        return await provider.complete(model=model, messages=messages, tools=tools)
