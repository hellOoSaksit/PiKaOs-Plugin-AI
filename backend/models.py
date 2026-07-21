"""AI / engine plugin models — the agent-runtime + LLM-connection schema this plugin OWNS.

`agents`, `runs`, `run_steps`, `stub_tool_writes`, `tasks`, `rooms`, `llm_connections`,
`llm_role_bindings` left Core in the AI extraction. They live on this plugin's OWN declarative `Base`
(separate metadata from the kernel), created by the plugin's migrate() step (create_all), never by
Core's Alembic baseline.

Cross-plugin refs are logical UUIDs, NOT foreign keys: `owner_id`/`created_by`/`department_id` point at
the auth plugin's users/departments by id with no DB-level FK. Foreign keys are kept only BETWEEN this
plugin's own tables (runs→agents/tasks/rooms, run_steps→runs, stub_tool_writes→runs,
llm_role_bindings→llm_connections).
"""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import BigInteger, Boolean, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """This plugin's declarative base — metadata independent of the kernel's `app.core.db.Base`."""


class LlmConnection(Base):
    """Runtime-configurable LLM provider (no-hardcode) — an admin sets provider/model/endpoint/key
    from the UI instead of `.env`. The API key is stored **encrypted** (app/crypto.py), never
    plaintext. At most one row is `is_active`; the worker resolves the active one per call so edits
    apply without a restart. This is the AI 3-mode connection model (Local/Remote/External)."""

    __tablename__ = "llm_connections"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    provider: Mapped[str] = mapped_column(String(32), nullable=False)  # ollama|openai|anthropic
    model: Mapped[str] = mapped_column(String(120), nullable=False, default="")
    base_url: Mapped[str | None] = mapped_column(String(512), nullable=True)
    api_key_enc: Mapped[str | None] = mapped_column(String(1024), nullable=True)  # Fernet ciphertext
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    # Last save-time connection test. A row is only persisted once its test passes, so this is "ok"
    # for anything created/edited under the test-on-save flow; NULL means "not tested yet" (a legacy
    # row from before the flow). Failures never persist, so "failed" is never stored here. The UI reads
    # this from the server (not local state) to render the status tag.
    last_test_status: Mapped[str | None] = mapped_column(String(16), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class LlmRoleBinding(Base):
    """Per-system LLM assignment (no-hardcode) — maps a system role (engine/search/summarize)
    to a specific `llm_connections` row, so an admin can route e.g. the search/RAG system to a
    local llama while the engine uses Claude. A role with no row falls back to the active
    connection (then the .env provider). FK cascades when the connection is deleted."""

    __tablename__ = "llm_role_bindings"

    role: Mapped[str] = mapped_column(String(32), primary_key=True)  # engine | search | summarize
    connection_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("llm_connections.id", ondelete="CASCADE"), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


# --- Engine (system-design §7; FK/index per risk-mitigation §4.4) ---


class Room(Base):
    __tablename__ = "rooms"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(120), nullable=False, default="")
    template: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    created_by: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    department_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class Agent(Base):
    """An AI agent. `status` is set by the runner only (product rule), never user-settable."""

    __tablename__ = "agents"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    owner_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False, default="")
    role: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="idle")
    model: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    skills: Mapped[list[str]] = mapped_column(ARRAY(String), nullable=False, default=list)
    granted_tools: Mapped[list[str]] = mapped_column(ARRAY(String), nullable=False, default=list)
    sprite: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    room_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("rooms.id", ondelete="SET NULL"), nullable=True
    )
    department_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class Task(Base):
    __tablename__ = "tasks"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title: Mapped[str] = mapped_column(String(255), nullable=False, default="")
    brief: Mapped[str] = mapped_column(Text, nullable=False, default="")
    room_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("rooms.id", ondelete="SET NULL"), nullable=True
    )
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="open")
    created_by: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    department_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class Run(Base):
    """An execution run — kind 'agent' (the §4 loop) or 'orchestration' (orchestrator)."""

    __tablename__ = "runs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    kind: Mapped[str] = mapped_column(String(16), nullable=False, default="agent")
    parent_run_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("runs.id", ondelete="CASCADE"), nullable=True
    )
    agent_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("agents.id", ondelete="SET NULL"), nullable=True
    )
    task_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("tasks.id", ondelete="SET NULL"), nullable=True
    )
    room_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("rooms.id", ondelete="SET NULL"), nullable=True
    )
    department_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="queued")
    input: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    tokens_used: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class RunStep(Base):
    """One worklog step. tool steps are two-phase (pending→done) with a deterministic
    idempotency_key for replay-safe resume (risk-mitigation §1). UNIQUE(run_id, seq)."""

    __tablename__ = "run_steps"
    __table_args__ = (UniqueConstraint("run_id", "seq", name="uq_run_steps_run_seq"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("runs.id", ondelete="CASCADE"), nullable=False
    )
    seq: Mapped[int] = mapped_column(Integer, nullable=False)
    kind: Mapped[str] = mapped_column(String(16), nullable=False)              # llm|tool|message|status
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="done")  # pending|done|failed
    idempotency_key: Mapped[str | None] = mapped_column(String(128), nullable=True)
    role: Mapped[str | None] = mapped_column(String(32), nullable=True)
    content: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class StubToolWrite(Base):
    """Sink for the B4 stub tools. Lets tests observe the engine's two-phase / effect-class semantics
    (at-most-once side_effect vs deduped idempotent_write). UNIQUE(idempotency_key) backs the upsert
    tool's ON CONFLICT DO NOTHING."""

    __tablename__ = "stub_tool_writes"
    __table_args__ = (UniqueConstraint("idempotency_key", name="uq_stub_tool_writes_key"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("runs.id", ondelete="CASCADE"), index=True, nullable=True
    )
    tool: Mapped[str] = mapped_column(String(120), nullable=False, default="")
    idempotency_key: Mapped[str] = mapped_column(String(128), nullable=False)
    payload: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
