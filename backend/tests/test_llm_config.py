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
