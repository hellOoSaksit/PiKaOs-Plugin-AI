"""Real LLM adapter — local Ollama (C1, first slice).

Implements the same `LLMProvider` interface the B4 stub does
(`complete(model, messages, tools) -> LLMResult`), so the worker swaps it in via
`agent_runner.set_engine_runtime` with **zero changes to the agent loop**. The provider is
where vendor shape lives; the loop stays vendor-agnostic.

Talks to Ollama's native chat endpoint (`POST {base}/api/chat`, `stream=false`) over
**httpx** — already a backend dependency, so C1 adds **no new package** (tech-stack §3.2).
OpenAI / Anthropic adapters land next behind the same interface; rate-limit + backoff is C2.

The translation is three pure functions (`to_ollama_messages` / `to_ollama_tools` /
`parse_ollama_response`) so the mapping is unit-tested without a network — `complete()` is a
thin httpx shim over them.

Message shapes (the loop's internal format ↔ Ollama):
  internal assistant {"role":"assistant","content","tool_call":{"name","args"}}
      → ollama {"role":"assistant","content","tool_calls":[{"function":{"name","arguments"}}]}
  internal tool      {"role":"tool","name","content":<any>}  → ollama {"role":"tool","content":<json str>}
"""
from __future__ import annotations

import json

import httpx

from ...core.config import settings
from .agent_runner import LLMResult

# Agents are seeded with model "stub" (B4); when running a real provider treat that as
# "unset" and fall back to the configured default model rather than asking Ollama for "stub".
_UNSET_MODELS = {"", "stub"}


def _as_text(content) -> str:
    """Ollama wants string content; JSON-encode anything structured (tool results, etc.)."""
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    return json.dumps(content, ensure_ascii=False, default=str)


def to_ollama_messages(messages: list[dict]) -> list[dict]:
    """Map the agent loop's conversation to Ollama's `messages` array."""
    out: list[dict] = []
    for m in messages:
        role = m.get("role") or "user"
        if role == "assistant":
            am: dict = {"role": "assistant", "content": _as_text(m.get("content", ""))}
            tc = m.get("tool_call")
            if tc:
                am["tool_calls"] = [{"function": {"name": tc.get("name", ""), "arguments": tc.get("args") or {}}}]
            out.append(am)
        elif role == "tool":
            out.append({"role": "tool", "content": _as_text(m.get("content"))})
        else:  # user / system
            out.append({"role": role, "content": _as_text(m.get("content", ""))})
    return out


def to_ollama_tools(tools: list[dict]) -> list[dict]:
    """Map the tool registry's schemas to Ollama function specs.

    Stub tools carry only name + description (no params) → default to an empty object
    schema. A tool that declares its own `parameters` JSON-schema passes through.
    """
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


def parse_ollama_response(data: dict) -> LLMResult:
    """Normalize one `/api/chat` response into an `LLMResult` (vendor tool-use flattened)."""
    msg = (data or {}).get("message") or {}
    text = msg.get("content") or ""
    # tokens charged to the user's quota = prompt + completion (both cost the model).
    tokens = int(data.get("prompt_eval_count") or 0) + int(data.get("eval_count") or 0)

    calls = msg.get("tool_calls") or []
    if calls:
        fn = (calls[0] or {}).get("function") or {}
        args = fn.get("arguments")
        if isinstance(args, str):  # OpenAI-compat returns a JSON string; native returns a dict
            try:
                args = json.loads(args)
            except ValueError:
                args = {}
        return LLMResult(text=text, stop_reason="tool_use", tokens=tokens,
                         tool_name=fn.get("name"), tool_args=args or {})
    return LLMResult(text=text, stop_reason="end", tokens=tokens)


class OllamaProvider:
    """LLMProvider backed by a local Ollama server. See module docstring."""

    def __init__(self, *, base_url: str | None = None, model: str | None = None,
                 timeout: float | None = None, transport: httpx.BaseTransport | None = None):
        self.base_url = (base_url or settings.llm_base_url).rstrip("/")
        self.default_model = model or settings.llm_default_model
        self.timeout = timeout or settings.llm_request_timeout_s
        self._transport = transport  # tests inject httpx.MockTransport; None → real network

    async def complete(self, *, model: str, messages: list[dict], tools: list[dict]) -> LLMResult:
        chosen = model if model not in _UNSET_MODELS else self.default_model
        body: dict = {
            "model": chosen,
            "messages": to_ollama_messages(messages),
            "stream": False,
        }
        ollama_tools = to_ollama_tools(tools)
        if ollama_tools:
            body["tools"] = ollama_tools

        async with httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout,
                                     transport=self._transport) as client:
            resp = await client.post("/api/chat", json=body)
            resp.raise_for_status()
            return parse_ollama_response(resp.json())
