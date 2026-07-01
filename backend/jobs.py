"""The arq job(s) this plugin contributes to the worker. Moved from Core `worker.py` in the AI
extraction — the worker collects these via the Loader (`collect_jobs`) and no longer imports the engine."""
from __future__ import annotations

from . import agent_runner


async def agent_run(ctx, run_id: str) -> str:
    """arq job: execute (or resume) one agent run. See agent_runner.run_job.

    Keyed by run_id so arq dedups concurrent enqueues (resume replay-safety) — the `keep_result`
    wrapper is applied where this is registered (`__init__.jobs`)."""
    return await agent_runner.run_job(run_id)
