"""WebSocket first-message auth (A2).

Network-free: drives `ws._authenticate` with a one-shot fake socket. Proves the token is taken from the
first frame (not the URL) and delegated to the identity provider resolved from `websocket.app` — the
provider (auth plugin) owns JWT decode + denylist + status checks (tested in test_auth / test_identity_*);
ws only extracts the token, delegates, and maps the result to a user_id string or None.

    docker compose exec backend pytest tests/test_ws.py
"""
from __future__ import annotations

import json
from types import SimpleNamespace

from app.core import contracts
from app.plugins.ai import db_ref
from app.plugins.ai import ws as wsmod


class _NoopSession:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


# _can_view_task opens a session before delegating to task_service.can_view, which rejects non-UUID ids
# before any query. The plugin lifecycle (which binds db_ref from postgres.Connection) doesn't run in this
# unit test, so bind a no-op factory — the malformed-id path never actually touches the DB.
db_ref.bind({"session_factory": lambda: _NoopSession()})


class _FakeProvider:
    """Accepts exactly one token, returning a user; everything else (garbage / denied / expired) → None,
    exactly as the real AuthIdentityProvider collapses those cases."""

    def __init__(self, ok_token: str | None = None):
        self._ok = ok_token

    async def authenticate(self, token):
        return SimpleNamespace(id="u1", role="member", status="active") if token and token == self._ok else None


class _OneShotWS:
    """Fake WebSocket: first `receive_text` returns a preset frame; `.app` carries a DI container whose
    IDENTITY resolves to the given provider (the deny-all bootstrap when None)."""

    def __init__(self, text: str, provider=None):
        self._text = text
        container = SimpleNamespace(resolve=lambda k: provider if k == contracts.IDENTITY else None)
        self.app = SimpleNamespace(state=SimpleNamespace(container=container))

    async def receive_text(self) -> str:
        return self._text


async def test_first_message_auth_accepts_valid_token():
    ws = _OneShotWS(json.dumps({"type": "auth", "token": "good"}), provider=_FakeProvider("good"))
    assert await wsmod._authenticate(ws) == "u1"


async def test_rejects_when_first_frame_is_not_auth():
    ws = _OneShotWS(json.dumps({"type": "subscribe", "task_id": "q1"}), provider=_FakeProvider("good"))
    assert await wsmod._authenticate(ws) is None


async def test_rejects_garbage_token():
    ws = _OneShotWS(json.dumps({"type": "auth", "token": "not-a-jwt"}), provider=_FakeProvider("good"))
    assert await wsmod._authenticate(ws) is None


async def test_rejects_non_json_first_frame():
    ws = _OneShotWS("hello", provider=_FakeProvider("good"))
    assert await wsmod._authenticate(ws) is None


async def test_rejects_when_provider_denies_token():
    # A denied / expired token is one the provider rejects (returns None) — ws maps that to None.
    ws = _OneShotWS(json.dumps({"type": "auth", "token": "denied"}), provider=_FakeProvider("good"))
    assert await wsmod._authenticate(ws) is None


async def test_no_provider_bound_denies():
    # auth plugin off → IDENTITY unbound → BootstrapProvider denies → None.
    ws = _OneShotWS(json.dumps({"type": "auth", "token": "good"}), provider=None)
    assert await wsmod._authenticate(ws) is None


async def test_task_authz_denies_malformed_ids():
    # _can_view_task runs real authz (task_service.can_view); non-UUID ids are rejected before any DB hit.
    assert await wsmod._can_view_task("u1", "q1") is False
