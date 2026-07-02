"""DB session factory for the ai engine — resolved from the `postgres.Connection` DI contract.

The zero-datastore kernel owns no engine (no `app.core.db.SessionLocal`): the postgres Tool creates the
engine + session factory and binds them under `postgres.Connection`. This plugin's register() resolves
that contract and stashes the factory here; the run loop / ws / stub tools / llm-config service open
sessions via `new_session()` at call time (never `from . import _sf`) so they see the bound factory.
`postgres` is a declared dependency, so it is always bound when ai is enabled — mirrors `redis_bus.bind`.
"""
from __future__ import annotations

_sf = None  # async_sessionmaker, bound from postgres.Connection at register()


def bind(conn) -> None:
    """Wire the session factory from the postgres.Connection contract (called by the plugin's register())."""
    global _sf
    _sf = conn["session_factory"] if conn else None


def new_session():
    """Open a new AsyncSession (async context manager) from the bound factory. Raises if postgres is
    unbound — a declared dependency, so this only fires on a genuine misconfiguration."""
    if _sf is None:
        raise RuntimeError("ai: postgres.Connection not bound — enable the postgres tool")
    return _sf()
