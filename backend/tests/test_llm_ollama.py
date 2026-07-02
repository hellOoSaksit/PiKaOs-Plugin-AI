"""Tests for the C1 Ollama LLM adapter.

The mapping logic lives in three pure functions (no network) → driven directly.
`complete()` is exercised end-to-end against an httpx MockTransport, so the real httpx
request/response path runs without an Ollama server.

    docker compose exec backend pytest tests/test_llm_ollama.py
"""
from __future__ import annotations

import asyncio
import json

import httpx

from app.plugins.ai.llm_ollama import (
    OllamaProvider,
    parse_ollama_response,
    to_ollama_messages,
    to_ollama_tools,
)


# --- to_ollama_messages -----------------------------------------------------


def test_messages_map_each_role():
    out = to_ollama_messages([
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "calling", "tool_call": {"name": "echo", "args": {"x": 1}}},
        {"role": "tool", "name": "echo", "content": {"echo": {"x": 1}}},
    ])
    assert out[0] == {"role": "user", "content": "hi"}
    assert out[1]["role"] == "assistant" and out[1]["content"] == "calling"
    assert out[1]["tool_calls"] == [{"function": {"name": "echo", "arguments": {"x": 1}}}]
    # tool result (a dict) is JSON-encoded to a string for Ollama
    assert out[2]["role"] == "tool" and json.loads(out[2]["content"]) == {"echo": {"x": 1}}


def test_assistant_without_tool_call_has_no_tool_calls_key():
    out = to_ollama_messages([{"role": "assistant", "content": "done"}])
    assert "tool_calls" not in out[0]


# --- to_ollama_tools --------------------------------------------------------


def test_tools_map_to_function_specs_with_default_params():
    out = to_ollama_tools([{"name": "echo", "description": "d"}])
    assert out == [{
        "type": "function",
        "function": {"name": "echo", "description": "d", "parameters": {"type": "object", "properties": {}}},
    }]


def test_tools_passthrough_declared_parameters_and_skip_nameless():
    schema = {"type": "object", "properties": {"q": {"type": "string"}}}
    out = to_ollama_tools([{"name": "search", "parameters": schema}, {"description": "no name"}])
    assert len(out) == 1 and out[0]["function"]["parameters"] == schema


def test_empty_tools_is_empty():
    assert to_ollama_tools([]) == []


# --- parse_ollama_response --------------------------------------------------


def test_parse_final_answer_sums_tokens():
    r = parse_ollama_response({
        "message": {"role": "assistant", "content": "hi there"},
        "prompt_eval_count": 10, "eval_count": 5,
    })
    assert r.stop_reason == "end" and r.text == "hi there" and r.tokens == 15
    assert r.tool_name is None


def test_parse_tool_use():
    r = parse_ollama_response({
        "message": {"role": "assistant", "content": "",
                    "tool_calls": [{"function": {"name": "echo", "arguments": {"x": 1}}}]},
        "eval_count": 4,
    })
    assert r.stop_reason == "tool_use" and r.tool_name == "echo" and r.tool_args == {"x": 1} and r.tokens == 4


def test_parse_tool_args_as_json_string():
    r = parse_ollama_response({
        "message": {"tool_calls": [{"function": {"name": "f", "arguments": "{\"a\": 2}"}}]},
    })
    assert r.tool_args == {"a": 2}


def test_parse_missing_fields_safe():
    r = parse_ollama_response({})
    assert r.stop_reason == "end" and r.text == "" and r.tokens == 0


# --- complete() over a mocked transport -------------------------------------


def _mock(captured: dict, response: dict):
    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=response)
    return httpx.MockTransport(handler)


def test_complete_posts_to_api_chat_and_parses():
    captured: dict = {}
    transport = _mock(captured, {
        "message": {"role": "assistant", "content": "summary"},
        "prompt_eval_count": 7, "eval_count": 3,
    })
    prov = OllamaProvider(base_url="http://ollama:11434", model="llama3.1", transport=transport)
    r = asyncio.run(prov.complete(
        model="", messages=[{"role": "user", "content": "sum it"}],
        tools=[{"name": "echo", "description": "d"}],
    ))
    assert r.text == "summary" and r.tokens == 10
    assert captured["url"].endswith("/api/chat")
    assert captured["body"]["model"] == "llama3.1"      # empty/"stub" model → configured default
    assert captured["body"]["stream"] is False
    assert captured["body"]["tools"][0]["function"]["name"] == "echo"


def test_complete_omits_tools_when_none():
    captured: dict = {}
    transport = _mock(captured, {"message": {"content": "ok"}})
    prov = OllamaProvider(base_url="http://ollama:11434", model="m", transport=transport)
    asyncio.run(prov.complete(model="custom", messages=[{"role": "user", "content": "x"}], tools=[]))
    assert "tools" not in captured["body"]
    assert captured["body"]["model"] == "custom"        # explicit model wins over default
