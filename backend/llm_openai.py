"""Real LLM adapter — OpenAI / ChatGPT Chat Completions API (C1).

Same `LLMProvider` interface as the stub / Ollama / Anthropic adapters; talks to
`POST {base}/chat/completions` over **httpx** (no `openai` SDK added — zero-extra-dep policy).
Works against any OpenAI-compatible endpoint (set `OPENAI_BASE_URL`).

Wire shape:
  - header `Authorization: Bearer <key>`
  - request `{model, messages, tools, max_tokens}` — `system` is a message with role "system"
  - tool calls: `choices[0].message.tool_calls[0].function.{name, arguments(JSON string)}`;
    a tool result is a `{role:"tool", tool_call_id, content}` message paired to that call id
  - tokens = `usage.total_tokens` (fallback to prompt+completion)
"""
from __future__ import annotations

import json

import httpx

from ...core.config import settings
from .agent_runner import LLMResult

_UNSET_MODELS = {"", "stub"}


def _as_text(content) -> str:
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    return json.dumps(content, ensure_ascii=False, default=str)


def to_openai_messages(messages: list[dict]) -> list[dict]:
    """Map the loop conversation to OpenAI's `messages` array (tool ids threaded)."""
    out: list[dict] = []
    counter = 0
    pending_id: str | None = None

    for m in messages:
        role = m.get("role") or "user"
        if role == "assistant":
            msg: dict = {"role": "assistant", "content": _as_text(m.get("content", "")) or None}
            tc = m.get("tool_call")
            if tc:
                pending_id = f"call_{counter}"
                counter += 1
                msg["tool_calls"] = [{
                    "id": pending_id, "type": "function",
                    "function": {"name": tc.get("name", ""),
                                 "arguments": json.dumps(tc.get("args") or {}, ensure_ascii=False)},
                }]
            out.append(msg)
        elif role == "tool":
            tid = pending_id or f"call_{counter}"
            out.append({"role": "tool", "tool_call_id": tid, "content": _as_text(m.get("content"))})
            pending_id = None
        else:  # user / system
            out.append({"role": role, "content": _as_text(m.get("content", ""))})
    return out


def to_openai_tools(tools: list[dict]) -> list[dict]:
    """Map tool-registry schemas to OpenAI function specs (default empty object schema)."""
    out: list[dict] = []
    for t in tools or []:
        name = t.get("name")
        if not name:
            continue
        out.append({
            "type": "function",
            "function": {
                "name": name,
                "description": t.get("description", ""),
                "parameters": t.get("parameters") or {"type": "object", "properties": {}},
            },
        })
    return out


def parse_openai_response(data: dict) -> LLMResult:
    """Normalize a Chat Completions response into an `LLMResult` (first tool call flattened)."""
    choice = ((data or {}).get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    text = msg.get("content") or ""

    usage = data.get("usage") or {}
    tokens = int(usage.get("total_tokens") or 0) or (
        int(usage.get("prompt_tokens") or 0) + int(usage.get("completion_tokens") or 0))

    calls = msg.get("tool_calls") or []
    if calls:
        fn = (calls[0] or {}).get("function") or {}
        args = fn.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except ValueError:
                args = {}
        return LLMResult(text=text, stop_reason="tool_use", tokens=tokens,
                         tool_name=fn.get("name"), tool_args=args or {})
    return LLMResult(text=text, stop_reason="end", tokens=tokens)


class OpenAIProvider:
    """LLMProvider backed by the OpenAI (or OpenAI-compatible) Chat Completions API."""

    def __init__(self, *, api_key: str | None = None, base_url: str | None = None,
                 model: str | None = None, max_tokens: int | None = None,
                 timeout: float | None = None, transport: httpx.BaseTransport | None = None):
        self.api_key = api_key if api_key is not None else settings.openai_api_key
        self.base_url = (base_url or settings.openai_base_url).rstrip("/")
        self.default_model = model or settings.openai_default_model
        self.max_tokens = max_tokens or settings.llm_max_tokens
        self.timeout = timeout or settings.llm_request_timeout_s
        self._transport = transport

    async def complete(self, *, model: str, messages: list[dict], tools: list[dict]) -> LLMResult:
        chosen = model if model not in _UNSET_MODELS else self.default_model
        body: dict = {
            "model": chosen,
            "messages": to_openai_messages(messages),
            "max_tokens": self.max_tokens,
        }
        oai_tools = to_openai_tools(tools)
        if oai_tools:
            body["tools"] = oai_tools

        headers = {"Authorization": f"Bearer {self.api_key}", "content-type": "application/json"}
        async with httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout,
                                     transport=self._transport) as client:
            resp = await client.post("/chat/completions", json=body, headers=headers)
            resp.raise_for_status()
            return parse_openai_response(resp.json())

    async def probe(self, *, timeout: float = 5.0) -> dict:
        """Cheap liveness/auth check for the Test-connection button — GET {root}/models (no token
        cost). base_url is the API ROOT (e.g. http://localhost:1234/v1), so /models sits beside
        /chat/completions. Returns a sanitized {ok, status, category, detail}."""
        from . import llm_probe
        return await llm_probe.run_probe(
            method="GET", url=f"{self.base_url}/models",
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=timeout, transport=self._transport)
