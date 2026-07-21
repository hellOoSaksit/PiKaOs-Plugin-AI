"""Connection-probe helpers for the "Test connection" button — a cheap, keyed liveness check that an
admin's LLM connection (endpoint + key) actually works, without spending completion tokens.

SSRF stance (deliberately relaxed — see architecture/security.md): a `custom` connection's whole
purpose is a **self-run local LLM** (LM Studio at localhost, Ollama, host.docker.internal), so a normal
SSRF guard that blocks loopback/private would break the primary use case. We therefore ALLOW loopback +
private and only block the **cloud-metadata / link-local** range (169.254.0.0/16, fe80::/10) where no
real LLM lives — the one target worth denying. Configuring a base_url is already an `llm.manage`
capability the engine uses for real inference, so Test adds no new trust boundary; the incremental risk
is leaking an internal error body, which `categorize_*` prevents by returning only a sanitized class +
short message, never the raw response.
"""
from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse

import httpx


class BlockedURL(Exception):
    """base_url points at a blocked (metadata/link-local) host, or uses a non-http scheme."""


def assert_not_metadata(url: str) -> None:
    """Raise BlockedURL for a non-http(s) scheme or a host that resolves to a link-local/metadata
    address. Loopback + private are intentionally allowed (local LLMs)."""
    parts = urlparse(url or "")
    if parts.scheme not in ("http", "https"):
        raise BlockedURL(f"scheme not allowed: {parts.scheme or '(none)'}")
    host = parts.hostname
    if not host:
        raise BlockedURL("missing host")
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        # let the real probe surface an honest "can't connect"; DNS failure isn't a security block
        return
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_link_local:                       # 169.254.0.0/16 (cloud metadata) + fe80::/10
            raise BlockedURL(f"link-local / metadata address blocked: {host} -> {ip}")


# --- result categorization: sanitized (class + short message), never the raw body -----------------

_STATUS = {
    "auth": (401, 403),
    "not_found": (404,),
}


def categorize_status(status: int) -> tuple[str, str]:
    """Map an HTTP status to (category, short message). 2xx = ok."""
    if 200 <= status < 300:
        return "ok", ""
    if status in (401, 403):
        return "auth", "the API key was rejected"
    if status == 404:
        return "not_found", "endpoint or model not found — check the Base URL"
    if status == 429:
        return "rate_limit", "rate-limited by the provider"
    return "http", f"HTTP {status}"


def categorize_error(exc: Exception) -> tuple[str, str]:
    """Map an httpx transport error to (category, short message) — no raw internals leaked."""
    if isinstance(exc, httpx.TimeoutException):
        return "timeout", "the endpoint did not respond in time"
    if isinstance(exc, httpx.ConnectError):
        return "connection", "could not connect to the endpoint"
    if isinstance(exc, httpx.RequestError):
        return "connection", "network error reaching the endpoint"
    return "error", "unexpected error"


async def run_probe(*, method: str, url: str, headers: dict, timeout: float,
                    transport: httpx.BaseTransport | None = None) -> dict:
    """Do the cheap request and return a sanitized {ok, status, category, detail}. Never raises for a
    normal transport/HTTP failure — those become a category. Auth/keys never appear in the result."""
    try:
        async with httpx.AsyncClient(timeout=timeout, transport=transport) as client:
            resp = await client.request(method, url, headers=headers)
    except Exception as exc:                       # noqa: BLE001 — every failure becomes a category
        category, detail = categorize_error(exc)
        return {"ok": False, "status": None, "category": category, "detail": detail}
    category, detail = categorize_status(resp.status_code)
    return {"ok": category == "ok", "status": resp.status_code, "category": category, "detail": detail}
