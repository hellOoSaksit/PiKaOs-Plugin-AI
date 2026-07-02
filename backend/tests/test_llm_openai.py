"""Tests for the C1 OpenAI / ChatGPT Chat Completions adapter.

Pure functions driven directly; `complete()` runs against an httpx MockTransport.

    docker compose exec backend pytest tests/test_llm_openai.py
"""
from __future__ import annotations

import asyncio
import json

import httpx

from app.plugins.ai.llm_openai import (
    OpenAIProvider,
    parse_openai_response,
    to_openai_messages,
    to_openai_tools,
)


# --- to_openai_messages (tool-call id threading) ----------------------------


def test_messages_map_with_tool_call_threading():
    out = to_openai_messages([
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "calling", "tool_call": {"name": "echo", "args": {"x": 1}}},
        {"role": "tool", "name": "echo", "content": {"echo": {"x": 1}}},
    ])
    assert out[0] == {"role": "system", "content": "sys"}
    assert out[1] == {"role": "user", "content": "hi"}
    call = out[2]["tool_calls"][0]
    assert call["type"] == "function" and call["function"]["name"] == "echo"
    assert json.loads(call["function"]["arguments"]) == {"x": 1}   # arguments serialized to a string
    # the tool result references the same call id
    assert out[3]["role"] == "tool" and out[3]["tool_call_id"] == call["id"]


def test_assistant_without_tool_has_no_tool_calls():
    out = to_openai_messages([{"role": "assistant", "content": "done"}])
    assert "tool_calls" not in out[0] and out[0]["content"] == "done"


# --- to_openai_tools --------------------------------------------------------


def test_tools_map_to_function_specs():
    assert to_openai_tools([{"name": "echo", "description": "d"}]) == [
        {"type": "function", "function": {
            "name": "echo", "description": "d", "parameters": {"type": "object", "properties": {}}}},
    ]


# --- parse_openai_response --------------------------------------------------


def test_parse_text_uses_total_tokens():
    r = parse_openai_response({
        "choices": [{"message": {"role": "assistant", "content": "hello"}}],
        "usage": {"total_tokens": 11},
    })
    assert r.stop_reason == "end" and r.text == "hello" and r.tokens == 11


def test_parse_tool_call_parses_json_arguments():
    r = parse_openai_response({
        "choices": [{"message": {"tool_calls": [
            {"id": "call_0", "type": "function",
             "function": {"name": "echo", "arguments": "{\"x\": 1}"}}]}}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2},
    })
    assert r.stop_reason == "tool_use" and r.tool_name == "echo" and r.tool_args == {"x": 1}
    assert r.tokens == 5   # falls back to prompt+completion when total_tokens absent


# --- complete() over a mocked transport -------------------------------------


def _mock(captured: dict, response: dict):
    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["headers"] = request.headers
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=response)
    return httpx.MockTransport(handler)


def test_complete_posts_chat_completions_with_auth():
    captured: dict = {}
    transport = _mock(captured, {
        "choices": [{"message": {"content": "summary"}}],
        "usage": {"total_tokens": 9},
    })
    prov = OpenAIProvider(api_key="sk-oai", base_url="https://api.openai.com/v1",
                          model="gpt-4o-mini", max_tokens=512, transport=transport)
    r = asyncio.run(prov.complete(model="stub", messages=[{"role": "user", "content": "x"}], tools=[]))
    assert r.text == "summary" and r.tokens == 9
    assert captured["url"].endswith("/chat/completions")
    assert captured["headers"]["authorization"] == "Bearer sk-oai"
    assert captured["body"]["model"] == "gpt-4o-mini"   # "stub" treated as unset → default
    assert captured["body"]["max_tokens"] == 512
    assert "tools" not in captured["body"]
