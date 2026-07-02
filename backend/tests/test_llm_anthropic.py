"""Tests for the C1 Anthropic / Claude Messages adapter.

Pure translation/parse functions are driven directly; `complete()` runs against an httpx
MockTransport so the real request path (headers, URL, body) is exercised without a key.

    docker compose exec backend pytest tests/test_llm_anthropic.py
"""
from __future__ import annotations

import asyncio
import json

import httpx

from app.plugins.ai.llm_anthropic import (
    AnthropicProvider,
    parse_anthropic_response,
    to_anthropic,
    to_anthropic_tools,
)


# --- to_anthropic (system hoist + tool_use/tool_result pairing) -------------


def test_system_hoisted_and_roles_mapped():
    system, msgs = to_anthropic([
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "calling", "tool_call": {"name": "echo", "args": {"x": 1}}},
        {"role": "tool", "name": "echo", "content": {"echo": {"x": 1}}},
    ])
    assert system == "be terse"                          # system pulled to top level, not a message
    assert msgs[0] == {"role": "user", "content": "hi"}
    # assistant carries a text block + a tool_use block with a synthesized id
    blocks = msgs[1]["content"]
    assert blocks[0] == {"type": "text", "text": "calling"}
    tool_use = blocks[1]
    assert tool_use["type"] == "tool_use" and tool_use["name"] == "echo" and tool_use["input"] == {"x": 1}
    # the tool result is a USER turn referencing the SAME tool_use_id
    result = msgs[2]
    assert result["role"] == "user"
    assert result["content"][0]["type"] == "tool_result"
    assert result["content"][0]["tool_use_id"] == tool_use["id"]


def test_assistant_with_no_content_gets_nonempty_block():
    _, msgs = to_anthropic([{"role": "assistant", "content": ""}])
    assert msgs[0]["content"] == [{"type": "text", "text": ""}]  # Anthropic rejects empty content


def test_no_system_returns_none():
    system, _ = to_anthropic([{"role": "user", "content": "x"}])
    assert system is None


# --- to_anthropic_tools -----------------------------------------------------


def test_tools_map_with_default_input_schema():
    assert to_anthropic_tools([{"name": "echo", "description": "d"}]) == [
        {"name": "echo", "description": "d", "input_schema": {"type": "object", "properties": {}}},
    ]


# --- parse_anthropic_response ----------------------------------------------


def test_parse_text_sums_usage():
    r = parse_anthropic_response({
        "content": [{"type": "text", "text": "hello"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 8, "output_tokens": 4},
    })
    assert r.stop_reason == "end" and r.text == "hello" and r.tokens == 12 and r.tool_name is None


def test_parse_tool_use():
    r = parse_anthropic_response({
        "content": [
            {"type": "text", "text": "let me"},
            {"type": "tool_use", "id": "toolu_0", "name": "echo", "input": {"x": 1}},
        ],
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 3, "output_tokens": 2},
    })
    assert r.stop_reason == "tool_use" and r.tool_name == "echo" and r.tool_args == {"x": 1} and r.tokens == 5


def test_parse_refusal_is_end_with_empty_text():
    r = parse_anthropic_response({"content": [], "stop_reason": "refusal"})
    assert r.stop_reason == "end" and r.text == ""


# --- complete() over a mocked transport -------------------------------------


def _mock(captured: dict, response: dict):
    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["headers"] = request.headers
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=response)
    return httpx.MockTransport(handler)


def test_complete_sends_headers_body_and_parses():
    captured: dict = {}
    transport = _mock(captured, {
        "content": [{"type": "text", "text": "summary"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 5, "output_tokens": 5},
    })
    prov = AnthropicProvider(api_key="sk-test", base_url="https://api.anthropic.com",
                             model="claude-opus-4-8", max_tokens=1024, transport=transport)
    r = asyncio.run(prov.complete(
        model="", messages=[{"role": "user", "content": "sum"}],
        tools=[{"name": "echo", "description": "d"}],
    ))
    assert r.text == "summary" and r.tokens == 10
    assert captured["url"].endswith("/v1/messages")
    assert captured["headers"]["x-api-key"] == "sk-test"
    assert captured["headers"]["anthropic-version"] == "2023-06-01"
    assert captured["body"]["model"] == "claude-opus-4-8"   # empty model → configured default
    assert captured["body"]["max_tokens"] == 1024           # required by Anthropic
    assert "temperature" not in captured["body"]            # removed on Opus 4.8 (would 400)
    assert captured["body"]["tools"][0]["name"] == "echo"
