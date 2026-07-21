"""Tests for runtime LLM provider config (no-hardcode) — crypto, masking, provider build, DB activate.

    docker compose exec backend pytest tests/test_llm_config.py
"""
from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from types import SimpleNamespace

from sqlalchemy import delete as sql_delete
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core import crypto
from app.core.config import settings
from app.plugins.ai.models import LlmConnection, LlmRoleBinding
from app.plugins.ai import llm_connections_repo as repo
from app.plugins.ai import llm_role_bindings_repo as role_repo
from app.plugins.ai import llm_config_service as svc
from app.plugins.ai.llm_anthropic import AnthropicProvider
from app.plugins.ai.llm_ollama import OllamaProvider
from app.plugins.ai.llm_openai import OpenAIProvider


# --- crypto (encrypt at rest) -----------------------------------------------


def test_crypto_roundtrip():
    token = crypto.encrypt("sk-secret-123")
    assert token != "sk-secret-123"                    # actually encrypted, not stored raw
    assert crypto.decrypt(token) == "sk-secret-123"


def test_decrypt_bad_input_is_empty():
    assert crypto.decrypt("") == ""
    assert crypto.decrypt("not-a-valid-token") == ""   # never raises — returns ""


# --- masking + provider build (pure) ----------------------------------------


def _row(**kw):
    base = dict(id=uuid.uuid4(), name="c", provider="anthropic", model="claude-opus-4-8",
                base_url=None, api_key_enc=None, is_active=False,
                created_at=datetime.now(timezone.utc))
    base.update(kw)
    return SimpleNamespace(**base)


def test_to_out_masks_the_key():
    out = svc.to_out(_row(api_key_enc=crypto.encrypt("sk-x")))
    assert out["api_key_set"] is True
    assert "api_key" not in out and "api_key_enc" not in out   # raw key never leaves the service


def test_to_out_no_key():
    assert svc.to_out(_row(api_key_enc=None))["api_key_set"] is False


def test_build_provider_picks_type_by_provider():
    assert isinstance(svc.build_provider(_row(provider="anthropic", api_key_enc=crypto.encrypt("k"))), AnthropicProvider)
    assert isinstance(svc.build_provider(_row(provider="openai")), OpenAIProvider)
    assert isinstance(svc.build_provider(_row(provider="ollama")), OllamaProvider)


def test_create_rejects_unknown_provider():
    try:
        asyncio.run(svc.create(None, name="x", provider="bogus", model="", base_url=None, api_key=None))
        assert False, "expected BadProvider"
    except svc.BadProvider:
        pass


# --- per-system role assignment ---------------------------------------------


def test_set_role_rejects_unknown_role():
    # role is validated before any DB access, so db=None is fine here
    try:
        asyncio.run(svc.set_role(None, "bogus-role", None))
        assert False, "expected BadRole"
    except svc.BadRole:
        pass


def test_role_binding_resolves_and_clears():
    """set_binding → get_connection_for_role returns it; clear_binding → falls back to None."""
    cid = None

    async def main():
        nonlocal cid
        eng = create_async_engine(settings.database_url)
        Session = async_sessionmaker(eng, expire_on_commit=False, class_=AsyncSession)
        try:
            async with Session() as db:
                row = await repo.insert_connection(db, name="SearchLlama", provider="ollama",
                                                   model="llama3.1", base_url=None, api_key_enc=None)
                cid = row.id
            async with Session() as db:
                await role_repo.set_binding(db, "search", cid)
                bound = await role_repo.get_connection_for_role(db, "search")
                bindings = await role_repo.list_bindings(db)
                missing = await role_repo.get_connection_for_role(db, "summarize")
            async with Session() as db:
                await role_repo.clear_binding(db, "search")
                after = await role_repo.get_connection_for_role(db, "search")
            return bound.id, bindings.get("search"), missing, after
        finally:
            async with Session() as c:
                await c.execute(sql_delete(LlmRoleBinding).where(LlmRoleBinding.role.in_(["search", "summarize"])))
                await c.execute(sql_delete(LlmConnection).where(LlmConnection.id == cid))
                await c.commit()
            await eng.dispose()

    bound_id, listed, missing, after = asyncio.run(main())
    assert bound_id == cid                 # the role resolves to the connection it was bound to
    assert listed == cid                   # list_bindings reflects it
    assert missing is None                 # an unbound role resolves to nothing (→ active fallback)
    assert after is None                   # cleared binding no longer resolves


# --- DB: only one active connection (real DB) -------------------------------


def test_set_active_keeps_one_active():
    a, b = uuid.uuid4(), uuid.uuid4()

    async def main():
        eng = create_async_engine(settings.database_url)
        Session = async_sessionmaker(eng, expire_on_commit=False, class_=AsyncSession)
        try:
            async with Session() as db:
                ra = await repo.insert_connection(db, name="A", provider="ollama", model="", base_url=None, api_key_enc=None)
                rb = await repo.insert_connection(db, name="B", provider="anthropic", model="claude-opus-4-8", base_url=None, api_key_enc=None)
                a_id, b_id = ra.id, rb.id
            async with Session() as db:
                await repo.set_active(db, a_id)
                act1 = await repo.get_active(db)
                await repo.set_active(db, b_id)        # switching deactivates A
                act2 = await repo.get_active(db)
                rows = await repo.list_connections(db)
                n_active = sum(1 for r in rows if r.is_active and r.id in (a_id, b_id))
                return act1.id, act2.id, a_id, b_id, n_active
        finally:
            async with Session() as c:
                await c.execute(sql_delete(LlmConnection).where(LlmConnection.id.in_([a, b])))
                # also clean the freshly-created ids if different
                await c.commit()
            await eng.dispose()

    act1_id, act2_id, a_id, b_id, n_active = asyncio.run(main())
    assert act1_id == a_id and act2_id == b_id     # active follows set_active
    assert n_active == 1                            # never two active at once
    # teardown the rows we made (ids differ from the placeholder a/b)
    async def cleanup():
        eng = create_async_engine(settings.database_url)
        Session = async_sessionmaker(eng, expire_on_commit=False, class_=AsyncSession)
        async with Session() as c:
            await c.execute(sql_delete(LlmConnection).where(LlmConnection.id.in_([a_id, b_id])))
            await c.commit()
        await eng.dispose()
    asyncio.run(cleanup())


# --- custom (OpenAI-compatible) provider --------------------------------------


def test_build_provider_custom_uses_openai_adapter():
    row = _row(provider="custom", base_url="http://localhost:1234/v1/chat/completions",
               model="qwen2.5", api_key_enc=None)
    assert isinstance(svc.build_provider(row), OpenAIProvider)


def test_build_provider_custom_keyless_never_leaks_env_key(monkeypatch):
    """A keyless `custom` connection points at an admin-supplied endpoint (an arbitrary URL). Its
    built provider must NOT fall back to the server's .env OpenAI key — that would ship the real key
    to that endpoint (probe/inference `Authorization: Bearer <settings key>`). See CLAUDE.md rule 2."""
    monkeypatch.setattr(settings, "openai_api_key", "sk-env-secret")
    row = _row(provider="custom", base_url="http://untrusted.example/v1", api_key_enc=None)
    prov = svc.build_provider(row)
    assert prov.api_key != "sk-env-secret"   # empty/no key, not the .env fallback


def test_create_custom_requires_base_url():
    try:
        asyncio.run(svc.create(None, name="lm", provider="custom", model="",
                               base_url=None, api_key=None))
        assert False, "expected BadProvider"
    except svc.BadProvider:
        pass


def test_create_custom_with_base_url_ok(monkeypatch):
    async def fake_ins(db, **kw):
        return _row(provider="custom", base_url=kw["base_url"])
    monkeypatch.setattr(repo, "insert_connection", fake_ins)
    out = asyncio.run(svc.create(None, name="lm", provider="custom", model="qwen2.5",
                                 base_url="http://localhost:1234/v1/chat/completions",
                                 api_key=None))
    assert out["provider"] == "custom"


def test_update_to_custom_without_stored_base_url_rejected(monkeypatch):
    async def fake_get(db, cid):
        return _row(provider="ollama", base_url=None)
    monkeypatch.setattr(repo, "get_connection", fake_get)
    try:
        asyncio.run(svc.update(None, uuid.uuid4(), provider="custom"))
        assert False, "expected BadProvider"
    except svc.BadProvider:
        pass


def test_update_to_custom_with_stored_base_url_ok(monkeypatch):
    stored = _row(provider="ollama", base_url="http://localhost:11434/v1/chat/completions")

    async def fake_get(db, cid):
        return stored

    async def fake_upd(db, cid, **kw):
        return _row(provider="custom", base_url=stored.base_url)
    monkeypatch.setattr(repo, "get_connection", fake_get)
    monkeypatch.setattr(repo, "update_connection", fake_upd)
    out = asyncio.run(svc.update(None, uuid.uuid4(), provider="custom"))
    assert out["provider"] == "custom"


def test_update_clears_base_url_on_existing_custom_rejected(monkeypatch):
    stored = _row(provider="custom", base_url="http://localhost:1234/v1/chat/completions")

    async def fake_get(db, cid):
        return stored
    monkeypatch.setattr(repo, "get_connection", fake_get)
    try:
        asyncio.run(svc.update(None, uuid.uuid4(), base_url=""))   # provider omitted
        assert False, "expected BadProvider"
    except svc.BadProvider:
        pass


# --- manifest route declaration (mount gate) ----------------------------------
#
# UAT 2026-07-21 found the whole /api/llm surface silently unmounted: `routes: []` makes the
# loader treat the plugin as "no HTTP surface" (modules.active_modules skips load_router), and a
# declared route missing the `/ai` segment gets the plugin QUARANTINED by the §6 namespace gate.
# Unit tests can't see either failure (they import the routers directly), so pin the manifest.


def test_manifest_declares_namespaced_routes():
    import json
    from pathlib import Path

    manifest = json.loads((Path(__file__).parent.parent / "manifest.json").read_text(encoding="utf-8"))
    routes = manifest.get("routes", [])
    assert routes, "routes must be non-empty or the loader never mounts this plugin's router"
    for route in routes:
        assert "/ai" in route, f"route '{route}' must carry the /ai segment (§6) or the plugin is quarantined"


# --- role availability (owning module installed?) -----------------------------
#
# UAT 2026-07-21: the role panel offered binding for search/summarize/answer while the knowledge
# (RAG) plugin wasn't installed — misleading. roles_out now reports each role's consumer plugin +
# whether it is active, so the UI can disable the select and say "module not installed".


def test_roles_out_marks_unavailable_when_consumer_missing(monkeypatch):
    async def no_conns(db):
        return []

    async def no_bindings(db):
        return {}
    monkeypatch.setattr(repo, "list_connections", no_conns)
    monkeypatch.setattr(role_repo, "list_bindings", no_bindings)
    monkeypatch.setattr(svc, "_is_active", lambda pid: pid == "ai")   # knowledge NOT installed

    out = asyncio.run(svc.roles_out(None))
    by_role = {o["role"]: o for o in out}
    assert by_role["engine"]["plugin"] == "ai" and by_role["engine"]["available"] is True
    for r in ("search", "summarize", "answer"):
        assert by_role[r]["plugin"] == "knowledge"
        assert by_role[r]["available"] is False


def test_roles_out_available_when_consumer_active(monkeypatch):
    async def no_conns(db):
        return []

    async def no_bindings(db):
        return {}
    monkeypatch.setattr(repo, "list_connections", no_conns)
    monkeypatch.setattr(role_repo, "list_bindings", no_bindings)
    monkeypatch.setattr(svc, "_is_active", lambda pid: True)          # everything installed

    out = asyncio.run(svc.roles_out(None))
    assert all(o["available"] is True for o in out)
