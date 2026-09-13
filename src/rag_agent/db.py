from __future__ import annotations

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from rag_agent.config import get_settings


def _normalize_db_url(url: str) -> str:
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+asyncpg://", 1)
    return url


engine: AsyncEngine | None = None
SessionLocal: async_sessionmaker[AsyncSession] | None = None


def init_db() -> AsyncEngine:
    global engine, SessionLocal
    settings = get_settings()
    engine = create_async_engine(_normalize_db_url(settings.database_url), future=True)
    SessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    return engine


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    if SessionLocal is None:
        init_db()
    assert SessionLocal is not None
    async with SessionLocal() as session:
        yield session
