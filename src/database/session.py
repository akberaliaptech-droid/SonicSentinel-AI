"""Database session management and asynchronous SQLite engine for SonicSentinel AI.
"""
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy import select
from src.config import DB_PATH, MANDATORY_CLASSES, SEVERITY_MAPPING
from src.database.models import Base, Category, ModelVersion

DATABASE_URL = f"sqlite+aiosqlite:///{DB_PATH.as_posix()}"

engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    future=True,
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency for obtaining async DB sessions."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


async def init_db() -> None:
    """Initialize database tables and seed mandatory classes if not present."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with AsyncSessionLocal() as session:
        # Check if categories seeded
        result = await session.execute(select(Category))
        existing_categories = result.scalars().all()
        existing_names = {c.name for c in existing_categories}

        for class_name in MANDATORY_CLASSES:
            if class_name not in existing_names:
                severity = SEVERITY_MAPPING.get(class_name, "MEDIUM")
                new_cat = Category(
                    name=class_name,
                    severity=severity,
                    python_sample_count=5,
                    gtm_sample_count=5,
                    is_active=True,
                )
                session.add(new_cat)

        await session.commit()
