import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import redis.asyncio as redis
from neo4j import AsyncGraphDatabase
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import text
from loguru import logger

from core.config import settings


class Base(DeclarativeBase):
    pass


class DatabaseManager:
    def __init__(self):
        self.postgres_engine = None
        self.postgres_session_factory = None
        self.neo4j_driver = None
        self.redis_client = None

    async def initialize(self):
        await self._init_postgres()
        await self._init_neo4j()
        await self._init_redis()
        
    async def _init_postgres(self):
        self.postgres_engine = create_async_engine(
            settings.get_database_url,
            echo=settings.debug,
            pool_pre_ping=True,
            pool_recycle=3600,
        )
        self.postgres_session_factory = async_sessionmaker(
            self.postgres_engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
        # Verify PostgreSQL connection during initialization
        async with self.postgres_session_factory() as session:
            await session.execute(text("SELECT 1"))
        
        # Create tables if they don't exist (using existing migration SQL)
        await self._ensure_tables_exist()

    async def _init_neo4j(self):
        self.neo4j_driver = AsyncGraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )
        await self.neo4j_driver.verify_connectivity()

    async def _init_redis(self):
        self.redis_client = redis.from_url(
            settings.get_redis_url,
            decode_responses=True,
            retry_on_timeout=True,
        )
        await self.redis_client.ping()

    @asynccontextmanager
    async def get_postgres_session(self) -> AsyncGenerator[AsyncSession, None]:
        async with self.postgres_session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()

    @asynccontextmanager
    async def get_neo4j_session(self):
        async with self.neo4j_driver.session() as session:
            yield session

    async def get_redis_client(self) -> redis.Redis:
        return self.redis_client

    async def close(self):
        if self.postgres_engine:
            await self.postgres_engine.dispose()
        if self.neo4j_driver:
            await self.neo4j_driver.close()
        if self.redis_client:
            await self.redis_client.close()

    async def health_check(self) -> dict:
        health = {
            "postgres": False,
            "neo4j": False,
            "redis": False,
        }

        try:
            async with self.get_postgres_session() as session:
                await session.execute(text("SELECT 1"))
            health["postgres"] = True
        except Exception:
            pass

        try:
            async with self.get_neo4j_session() as session:
                await session.run("RETURN 1")
            health["neo4j"] = True
        except Exception:
            pass

        try:
            await self.redis_client.ping()
            health["redis"] = True
        except Exception:
            pass

        return health

    async def _ensure_tables_exist(self):
        """Ensure database tables exist using SQLAlchemy metadata."""
        try:
            # Import models and their Base to register them
            from .models_orm import Base as ModelsBase, Document, Chunk, ChatSession, ChatMessage, ProcessingJob
            
            # Skip ProcessingJob table creation since it's handled by migrations
            # Create a temporary metadata with only the tables we want to create
            from sqlalchemy import MetaData
            temp_metadata = MetaData()
            
            # Only create non-processing_jobs tables
            for table in ModelsBase.metadata.tables.values():
                if table.name != 'processing_jobs':
                    table.tometadata(temp_metadata)
            
            # Create all tables except processing_jobs
            async with self.postgres_engine.begin() as conn:
                await conn.run_sync(temp_metadata.create_all)
                
            logger.info("Database tables ensured to exist (excluding processing_jobs)")
            
        except Exception as e:
            logger.error(f"Failed to ensure tables exist: {e}")
            # Don't raise here - let the migration system handle complex cases
            logger.warning("Table creation failed, assuming migrations will handle it")


db_manager = DatabaseManager()