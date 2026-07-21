"""Stub LLM provider + tool registry for the engine (B4).

These prove the B3 loop end-to-end without paying for a real LLM or firing real side
effects — they implement the same `LLMProvider` / `ToolRegistry` interfaces the real
adapters (C1) will. The worker wires them at startup via `agent_runner.set_engine_runtime`.

**StubLLMProvider** answers in sequence from a script carried *in the conversation*: a seed
message whose content starts with `@@stub@@` followed by a JSON list of step specs —
`[{"text": "...", "tool": "record", "args": {...}, "tokens": 5}, {"text": "done"}]`. The
turn index = number of assistant messages so far. A spec with `"tool"` → a `tool_use` step;
without → a final answer. No script (or script exhausted) → echo the last user turn and stop.
Keeping the script in `run.input["messages"]` means the provider stays signature-pure
(`complete(model, messages, tools)`) — exactly what a real adapter sees.

**StubToolRegistry** exposes three tools, one per effect class (risk-mitigation §1):
`echo` (read) · `upsert` (idempotent_write → deduped) · `record` (side_effect → at-most-once).
Both writing tools persist to `stub_tool_writes` so behaviour is observable in tests.
"""
from __future__ import annotations

import json
import logging
import uuid

from . import db_ref
from . import stub_tools_repo as stub_repo
from .agent_runner import (
    EFFECT_IDEMPOTENT_WRITE,
    EFFECT_READ,
    EFFECT_SIDE_EFFECT,
    LLMResult,
)

log = logging.getLogger("pikaos.engine.stub")

SCRIPT_SENTINEL = "@@stub@@"


def _find_script(messages: list[dict]) -> list[dict] | None:
    for m in messages:
        content = m.get("content")
        if isinstance(content, str) and content.startswith(SCRIPT_SENTINEL):
            try:
                parsed = json.loads(content[len(SCRIPT_SENTINEL):])
                return parsed if isinstance(parsed, list) else None
            except (ValueError, TypeError):
                return None
    return None


def _est_tokens(text: str) -> int:
    return max(1, len(text) // 4)


class StubLLMProvider:
    """Deterministic, free LLM stand-in (see module docstring)."""

    async def probe(self, *, timeout: float = 5.0) -> dict:
        """The stub has no endpoint — report ok so a Test on the .env-fallback provider is honest
        ('the built-in stub is reachable') rather than a confusing network error."""
        return {"ok": True, "status": None, "category": "ok", "detail": "built-in stub (no endpoint)"}

    async def complete(self, *, model: str, messages: list[dict], tools: list[dict]) -> LLMResult:
        turn = sum(1 for m in messages if m.get("role") == "assistant")
        script = _find_script(messages)
        if script is not None and turn < len(script):
            spec = script[turn]
            text = str(spec.get("text", ""))
            tokens = int(spec.get("tokens", _est_tokens(text)))
            if spec.get("tool"):
                return LLMResult(text=text, stop_reason="tool_use", tokens=tokens,
                                 tool_name=str(spec["tool"]), tool_args=dict(spec.get("args", {})))
            return LLMResult(text=text, stop_reason="end", tokens=tokens)

        # no script / exhausted → finalize by echoing the last user message
        last_user = next((m.get("content", "") for m in reversed(messages) if m.get("role") == "user"), "")
        text = f"stub: {last_user}"
        return LLMResult(text=text, stop_reason="end", tokens=_est_tokens(text))


class StubToolRegistry:
    """One tool per effect class, writing to `stub_tool_writes` so tests can observe."""

    _EFFECTS = {"echo": EFFECT_READ, "upsert": EFFECT_IDEMPOTENT_WRITE, "record": EFFECT_SIDE_EFFECT}

    def schemas(self) -> list[dict]:
        return [
            {"name": "echo", "description": "Return the args (read-only)."},
            {"name": "upsert", "description": "Idempotent write to the stub sink (deduped by key)."},
            {"name": "record", "description": "Side-effect write to the stub sink (at-most-once)."},
        ]

    def effect_of(self, name: str) -> str:
        return self._EFFECTS.get(name, EFFECT_SIDE_EFFECT)

    async def call(self, name: str, args: dict, *, idempotency_key: str) -> dict:
        if name == "echo":
            return {"echo": args}
        run_id = _run_id_of(idempotency_key)
        async with db_ref.new_session() as db:
            if name == "upsert":
                inserted = await stub_repo.upsert_write(db, run_id, name, idempotency_key, args)
                return {"upserted": inserted, "key": idempotency_key}
            if name == "record":
                await stub_repo.record_write(db, run_id, name, idempotency_key, args)
                return {"recorded": True, "key": idempotency_key}
        raise ValueError(f"unknown stub tool: {name}")


def _run_id_of(idempotency_key: str) -> uuid.UUID | None:
    """idempotency_key is '{run_id}:{seq}' — recover the run id for the FK (None if malformed)."""
    try:
        return uuid.UUID(idempotency_key.split(":", 1)[0])
    except (ValueError, IndexError):
        return None
