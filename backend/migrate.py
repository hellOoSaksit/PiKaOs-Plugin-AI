"""Install-time schema step for the AI/engine plugin.

The kernel migration runner (`scripts.migrate_plugins`) calls `migrate(engine, session_factory)` for
each enabled plugin after Core's Alembic baseline. AI owns its 8 tables on its own `Base` metadata
(models.py); here we just create them. No seed — the engine starts empty (agents/rooms/tasks are
user-created). No pgvector — the AI tables have no vector columns.

Functional/fresh-DB model — plain create_all, not a versioned Alembic history yet.
"""
from __future__ import annotations

from .models import Base


async def migrate(engine, session_factory) -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
