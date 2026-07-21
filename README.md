# PiKaOs-Plugin-AI

The agent-ops **engine** as a Core plugin (id `ai`): the agent run loop, its tables
(agents/runs/run_steps/stub_tool_writes/tasks/rooms), the DB-configured LLM connection model
(llm_connections/llm_role_bindings + provider adapters), the WebSocket run/task stream, and the
`agent_run` worker job. Drops into Core via `link-plugins.sh` (runs on Core's 5173/8000).

Exposes the `ai.LLM` DI contract (a factory over the configured LLM provider) that other plugins —
today `knowledge` (RAG answer/summarize) — consume without importing this plugin. Consumes the
optional `knowledge.Retriever` contract for RAG-in-the-loop.

Deferred (plugin-proper, later): split each LLM provider into its own Tool (Ollama sidecar / Anthropic
/ OpenAI); split `tasks` and `rooms` into their own plugins; namespace the routes under `/api/ai`.

The plugin now also ships its **first frontend** (`frontend/`): an admin-only "LLM Config" screen
(descriptor + `LlmConfig.jsx` + pure `LlmConfig.logic.js`) for managing LLM provider connections —
including a `custom` OpenAI-compatible provider (base URL required, API key optional) — with its own
i18n packs (en/th/ja). It drops into Core's frontend the same way the backend half does, via
`link-plugins.sh`, and stays entirely absent from the bundle when the plugin isn't linked.
