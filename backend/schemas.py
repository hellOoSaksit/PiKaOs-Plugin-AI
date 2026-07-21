"""AI plugin request/response schemas — the LLM-connection config surface (moved from Core schemas.py
in the AI extraction). Drives `/api/llm-config`."""
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
    created_at: datetime


# Per-system LLM assignment (which connection a role uses — engine/search/summarize)
class LlmRoleSet(BaseModel):
    connection_id: uuid.UUID | None = None    # null = clear the binding (fall back to active)


class LlmRoleOut(BaseModel):
    role: str
    connection_id: uuid.UUID | None = None
    connection_name: str | None = None
