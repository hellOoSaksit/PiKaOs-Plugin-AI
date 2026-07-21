"""AI plugin request/response schemas — the LLM-connection config surface (moved from Core schemas.py
in the AI extraction). Drives `/api/ai/llm/connections` + `/api/ai/llm/roles`."""
from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, Field


class LlmConnectionIn(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    provider: str = Field(description="ollama | openai | anthropic | custom")
    model: str = ""
    base_url: str | None = None
    api_key: str | None = None         # write-only — encrypted at rest, never returned


class LlmConnectionUpdate(BaseModel):
    name: str | None = None
    provider: str | None = None
    model: str | None = None
    base_url: str | None = None
    api_key: str | None = None         # omit/empty = leave the stored key unchanged


class LlmConnectionOut(BaseModel):
    id: uuid.UUID
    name: str
    provider: str
    model: str
    base_url: str | None = None
    is_active: bool
    api_key_set: bool                  # masked — true if a key is stored (the value is never sent)
    last_test_status: str | None = None  # "ok" (passed at save time) | null (not tested yet) — drives the UI tag
    created_at: datetime


# Per-system LLM assignment (which connection a role uses — engine/search/summarize)
class LlmRoleSet(BaseModel):
    connection_id: uuid.UUID | None = None    # null = clear the binding (fall back to active)


class LlmRoleOut(BaseModel):
    role: str
    connection_id: uuid.UUID | None = None
    connection_name: str | None = None
    plugin: str = "ai"                        # which plugin consumes this role (engine=ai, RAG trio=knowledge)
    available: bool = True                    # False = that plugin isn't installed — UI disables binding


# Test-connection probe result (sanitized — never carries the key or a raw response body)
class LlmTestOut(BaseModel):
    ok: bool
    category: str                             # ok | auth | not_found | rate_limit | timeout | connection | http | blocked | error
    detail: str = ""                          # short, human-readable — safe to show
    status: int | None = None                 # upstream HTTP status, when there was one
    latency_ms: int | None = None
