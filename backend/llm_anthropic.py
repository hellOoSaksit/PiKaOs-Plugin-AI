"""Real LLM adapter — Anthropic / Claude Messages API (C1).

Implements the same `LLMProvider` interface as the stub (B4) and the Ollama adapter, so the
worker swaps it in via `set_engine_runtime` with no change to the agent loop. Talks to
`POST {base}/v1/messages` over **httpx** (already a backend dep — no `anthropic` SDK added,
matching the zero-extra-dependency policy and the Ollama adapter's approach).

Reference (verified via the claude-api skill, 2026-06-16):
  - headers: `x-api-key`, `anthropic-version: 2023-06-01`, `content-type: application/json`
  - request: `{model, max_tokens (required), system (top-level, NOT a message role), messages, tools}`
  - **do not send `temperature`/`top_p`/`budget_tokens`** — removed on Opus 4.8/4.7 (400)
  - response `content` is a block list: `{type:"text", text}` / `{type:"tool_use", id, name, input}`
  - tool turns must pair: an assistant `tool_use` block is answered by a user `tool_result` block
    carrying the same `tool_use_id`. tokens = `usage.input_tokens + usage.output_tokens`.

The loop's internal format carries no Anthropic tool ids, so we synthesize stable ids
(`toolu_<n>`) and thread each assistant `tool_use` to the following `tool` result.
"""
from __future__ import annotations

import json

import httpx

from ...core.config import settings
from .agent_runner import LLMResult

# Agents seeded with model "stub" (B4) → treat as unset and use the configured default.
_UNSET_MODELS = {"", "stub"}


def _as_text(content) -> str:
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    return json.dumps(content, ensure_ascii=False, default=str)


def to_anthropic(messages: list[dict]) -> tuple[str | None, list[dict]]:
    """Map the loop conversation to Anthropic's (system, messages) shape.

    System turns are hoisted to the top-level `system` string. Assistant `tool_call`s become
    `tool_use` blocks with a synthesized id; the next `tool` message becomes a user
    `tool_result` block referencing that id (Anthropic pairs them by `tool_use_id`).
    """
    system_parts: list[str] = []
    out: list[dict] = []
    counter = 0
    pending_id: str | None = None

    for m in messages:
        role = m.get("role") or "user"
        if role == "system":
            system_parts.append(_as_text(m.get("content", "")))
        elif role == "assistant":
            blocks: list[dict] = []
            text = m.get("content")
            if text:
                blocks.append({"type": "text", "text": _as_text(text)})
            tc = m.get("tool_call")
            if tc:
                pending_id = f"toolu_{counter}"
                counter += 1
                blocks.append({"type": "tool_use", "id": pending_id,
                               "name": tc.get("name", ""), "input": tc.get("args") or {}})
            out.append({"role": "assistant", "content": blocks or [{"type": "text", "text": ""}]})
        elif role == "tool":
            tid = pending_id or f"toolu_{counter}"
            out.append({"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": tid, "content": _as_text(m.get("content"))},
            ]})
            pending_id = None
        else:  # user
            out.append({"role": "user", "content": _as_text(m.get("content", ""))})

    system = "\n\n".join(p for p in system_parts if p) or None
    return system, out


def to_anthropic_tools(tools: list[dict]) -> list[dict]:
    """Map tool-registry schemas to Anthropic tool specs (default to an empty object schema)."""
    out: list[dict] = []
    for t in tools or []:
        name = t.get("name")
        if not name:
            continue
        out.append({
            "name": name,
            "description": t.get("description", ""),
            "input_schema": t.get("input_schema") or {"type": "object", "properties": {}},
        })
    return out


def parse_anthropic_response(data: dict) -> LLMResult:
    """Normalize a `/v1/messages` response into an `LLMResult` (first tool_use flattened)."""
    content = (data or {}).get("content") or []
    stop = data.get("stop_reason")
    text = "".join(b.get("text", "") for b in content if b.get("type") == "text")
    tool = next((b for b in content if b.get("type") == "tool_use"), None)

    usage = data.get("usage") or {}
    tokens = int(usage.get("input_tokens") or 0) + int(usage.get("output_tokens") or 0)

    if tool is not None and stop == "tool_use":
        return LLMResult(text=text, stop_reason="tool_use", tokens=tokens,
                         tool_name=tool.get("name"), tool_args=tool.get("input") or {})
    # end_turn / max_tokens / refusal → final answer (refusal yields empty text)
    return LLMResult(text=text, stop_reason="end", tokens=tokens)


class AnthropicProvider:
    """LLMProvider backed by the Anthropic Messages API. See module docstring."""

    def __init__(self, *, api_key: str | None = None, base_url: str | None = None,
                 model: str | None = None, version: str | None = None,
                 max_tokens: int | None = None, timeout: float | None = None,
                 transport: httpx.BaseTransport | None = None):
        self.api_key = api_key if api_key is not None else settings.anthropic_api_key
        self.base_url = (base_url or settings.anthropic_base_url).rstrip("/")
        self.default_model = model or settings.anthropic_default_model
        self.version = version or settings.anthropic_version
        self.max_tokens = max_tokens or settings.llm_max_tokens
        self.timeout = timeout or settings.llm_request_timeout_s
        self._transport = transport

    async def complete(self, *, model: str, messages: list[dict], tools: list[dict]) -> LLMResult:
        chosen = model if model not in _UNSET_MODELS else self.default_model
        system, anth_messages = to_anthropic(messages)
        body: dict = {"model": chosen, "max_tokens": self.max_tokens, "messages": anth_messages}
        if system:
            body["system"] = system
        anth_tools = to_anthropic_tools(tools)
        if anth_tools:
            body["tools"] = anth_tools

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": self.version,
            "content-type": "application/json",
        }
        async with httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout,
                                     transport=self._transport) as client:
            resp = await client.post("/v1/messages", json=body, headers=headers)
            resp.raise_for_status()
            return parse_anthropic_response(resp.json())

    async def probe(self, *, timeout: float = 5.0) -> dict:
        """Cheap liveness/auth check for the Test-connection button — GET {base}/v1/models (no token
        cost). base_url is the Anthropic root (no /v1). Returns a sanitized {ok, status, category, detail}."""
        from . import llm_probe
        return await llm_probe.run_probe(
            method="GET", url=f"{self.base_url}/v1/models",
            headers={"x-api-key": self.api_key, "anthropic-version": self.version},
            timeout=timeout, transport=self._transport)
