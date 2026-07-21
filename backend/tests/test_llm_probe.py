"""Tests for the connection-probe (Test connection button) — SSRF-lite guard + result categorization
+ per-adapter probe over a mocked httpx transport (no network).

    docker compose exec backend pytest tests/test_llm_probe.py
"""
from __future__ import annotations

import asyncio

import httpx
import pytest

from app.plugins.ai import llm_probe as probe
from app.plugins.ai.llm_openai import OpenAIProvider
from app.plugins.ai.llm_anthropic import AnthropicProvider


# --- SSRF-lite: allow local (the whole point of a self-run LLM), block cloud metadata ----------


def test_metadata_ip_blocked():
    with pytest.raises(probe.BlockedURL):
        probe.assert_not_metadata("http://169.254.169.254/latest/meta-data/")


def test_loopback_and_private_allowed():
    # local LLMs live here — must NOT be blocked
    probe.assert_not_metadata("http://localhost:1234/v1")
    probe.assert_not_metadata("http://127.0.0.1:11434")
    probe.assert_not_metadata("http://192.168.1.50:1234/v1")


def test_public_https_allowed():
    probe.assert_not_metadata("https://api.openai.com/v1")


def test_bad_scheme_blocked():
    with pytest.raises(probe.BlockedURL):
        probe.assert_not_metadata("file:///etc/passwd")


# --- categorization: sanitized, never the raw body ------------------------------------------------


def test_categorize_status():
    assert probe.categorize_status(200)[0] == "ok"
    assert probe.categorize_status(401)[0] == "auth"
    assert probe.categorize_status(403)[0] == "auth"
    assert probe.categorize_status(404)[0] == "not_found"
    assert probe.categorize_status(500)[0] == "http"


# --- per-adapter probe over a mock transport ------------------------------------------------------


def _transport(handler):
    return httpx.MockTransport(handler)


def test_openai_probe_ok_hits_models_on_the_base_root():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json={"data": []})

    p = OpenAIProvider(api_key="sk-x", base_url="http://localhost:1234/v1", transport=_transport(handler))
    res = asyncio.run(p.probe(timeout=2))
    assert res["ok"] is True and res["category"] == "ok"
    assert seen["url"] == "http://localhost:1234/v1/models"      # root + /models, no double path
    assert seen["auth"] == "Bearer sk-x"


def test_openai_probe_auth_rejected():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"message": "invalid api key"}})

    p = OpenAIProvider(api_key="bad", base_url="http://localhost:1234/v1", transport=_transport(handler))
    res = asyncio.run(p.probe(timeout=2))
    assert res["ok"] is False and res["category"] == "auth" and res["status"] == 401


def test_openai_probe_connection_error_is_caught_not_raised():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    p = OpenAIProvider(api_key="x", base_url="http://localhost:9999/v1", transport=_transport(handler))
    res = asyncio.run(p.probe(timeout=2))
    assert res["ok"] is False and res["category"] == "connection"


def test_anthropic_probe_hits_v1_models_with_key_headers():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["key"] = request.headers.get("x-api-key")
        seen["ver"] = request.headers.get("anthropic-version")
        return httpx.Response(200, json={"data": []})

    p = AnthropicProvider(api_key="sk-ant", base_url="https://api.anthropic.com", transport=_transport(handler))
    res = asyncio.run(p.probe(timeout=2))
    assert res["ok"] is True
    assert seen["url"] == "https://api.anthropic.com/v1/models"
    assert seen["key"] == "sk-ant" and seen["ver"]


# --- service: test_connection (guard + delegate to the provider probe) ----------------------------


def test_service_test_connection_blocks_metadata_base_url(monkeypatch):
    from types import SimpleNamespace
    from app.plugins.ai import llm_config_service as svc

    async def fake_get(db, cid):
        return SimpleNamespace(provider="custom", base_url="http://169.254.169.254/v1",
                               model="", api_key_enc=None)
    monkeypatch.setattr(svc.repo, "get_connection", fake_get)

    out = asyncio.run(svc.test_connection(None, "cid"))
    assert out["ok"] is False and out["category"] == "blocked"


def test_service_test_connection_delegates_and_times(monkeypatch):
    from types import SimpleNamespace
    from app.plugins.ai import llm_config_service as svc

    async def fake_get(db, cid):
        return SimpleNamespace(provider="custom", base_url="http://localhost:1234/v1",
                               model="", api_key_enc=None)

    class FakeProvider:
        async def probe(self, *, timeout=5.0):
            return {"ok": True, "status": 200, "category": "ok", "detail": ""}
    monkeypatch.setattr(svc.repo, "get_connection", fake_get)
    monkeypatch.setattr(svc, "build_provider", lambda row: FakeProvider())

    out = asyncio.run(svc.test_connection(None, "cid"))
    assert out["ok"] is True and out["category"] == "ok"
    assert "latency_ms" in out and isinstance(out["latency_ms"], int)


def test_service_test_connection_missing_raises_notfound(monkeypatch):
    from app.plugins.ai import llm_config_service as svc

    async def none_get(db, cid):
        return None
    monkeypatch.setattr(svc.repo, "get_connection", none_get)
    try:
        asyncio.run(svc.test_connection(None, "cid"))
        assert False, "expected NotFound"
    except svc.NotFound:
        pass
