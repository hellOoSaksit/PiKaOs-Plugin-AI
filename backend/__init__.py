"""AI / engine plugin — the agent-ops run loop + its tables + the LLM connection model + the WS stream.

Moved out of Core (was the always-on "engine" Base module + `worker.py`'s `agent_run` job). This plugin
OWNS `agents/runs/run_steps/stub_tool_writes/tasks/rooms` and `llm_connections/llm_role_bindings` on its
own `Base` (models.py), created by `migrate.migrate()` (run by `scripts.migrate_plugins`).

Contracts (contracts.py):
  - **provides `ai.LLM`** — a factory (`ConfiguredLLMProvider`); call it with a role to get a provider
    exposing `complete(*, model, messages, tools)`. Consumers (knowledge RAG answer/summarize) resolve
    it without importing this plugin.
  - **consumes `knowledge.Retriever`** (optional) — resolved at `boot()`; None when knowledge is off.

Package surface the Loader looks for (plugin-architecture.md §5/§10):
  router    — mounted when enabled (WS `/ws` + `/api/llm-config`); aggregated below
  register  — binds `ai.LLM` into the DI container
  boot      — wires the engine runtime (LLM provider + stub tools + optional retriever)
  jobs      — the `agent_run` arq job the worker runs when this plugin is enabled
  migrate   — install-time schema step (create_all), run by scripts.migrate_plugins
"""
from arq import func
from fastapi import APIRouter

from .jobs import agent_run
from .llm_config import roles_router as _llm_roles_router
from .llm_config import router as _llm_router
from .ws import router as _ws_router

# One aggregated router the Loader mounts (the loader mounts a single `router` per plugin). Sub-router
# prefixes are preserved: `/ws` (WebSocket) + `/api/llm-config` (+ its roles router).
router = APIRouter()
router.include_router(_ws_router)
router.include_router(_llm_router)
router.include_router(_llm_roles_router)

# `agent_run` is keyed by run_id → arq dedups concurrent enqueues (resume replay-safety).
jobs = [func(agent_run, keep_result=3600)]


def register(ctx) -> None:
    """Bind the `ai.LLM` factory so other plugins resolve the configured LLM without importing us."""
    from ...core.contracts import AI_LLM
    from .llm_config_service import ConfiguredLLMProvider

    ctx.container.bind(AI_LLM, ConfiguredLLMProvider)


def boot(ctx) -> None:
    """Wire the engine runtime once the container is assembled (register() ran for every plugin): the
    DB-configured LLM provider + stub tools + the OPTIONAL `knowledge.Retriever` (None when knowledge is
    disabled — the engine then runs without RAG context). Runs in both the web and worker composition
    roots; the worker is where `agent_run` actually uses the wired runtime."""
    from ...core.contracts import RETRIEVER
    from . import agent_runner
    from .engine_stubs import StubToolRegistry
    from .llm_config_service import ConfiguredLLMProvider

    retriever = ctx.container.resolve(RETRIEVER)
    agent_runner.set_engine_runtime(ConfiguredLLMProvider(), StubToolRegistry(), retriever=retriever)


__all__ = ["router", "jobs", "register", "boot"]
