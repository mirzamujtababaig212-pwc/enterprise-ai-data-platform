from __future__ import annotations

import asyncio
import os

import pytest
from sqlalchemy import delete, create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.control_plane.persistence.models import MemoryItemRecord
from memory.context.builder import MemoryContextBuilder
from memory.service import MemoryService
from memory.stores.postgres import PostgreSQLMemoryStore


def _postgres_store() -> tuple[PostgreSQLMemoryStore | None, object | None]:
    if os.getenv("RUN_POSTGRES_INTEGRATION") != "1":
        return None, None

    host = os.getenv("POSTGRES_TEST_HOST", "localhost")
    port = os.getenv("POSTGRES_TEST_PORT", "5432")
    user = os.getenv("POSTGRES_TEST_USER", "postgres")
    password = os.getenv("POSTGRES_TEST_PASSWORD", "postgres")
    database = os.getenv("POSTGRES_TEST_DB", "vehicle_platform")

    engine = create_engine(
        f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}",
        pool_pre_ping=True,
        future=True,
    )

    session_factory = sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
    )

    return PostgreSQLMemoryStore(session_factory), engine


def _cleanup(engine: object, memory_ids: list[str]) -> None:
    session_factory = sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
    )
    session: Session = session_factory()

    try:
        session.execute(
            delete(MemoryItemRecord).where(
                MemoryItemRecord.id.in_(memory_ids),
            )
        )
        session.commit()
    finally:
        session.close()


def test_postgresql_memory_context_builder_round_trip() -> None:
    store, engine = _postgres_store()

    if store is None:
        pytest.skip("Set RUN_POSTGRES_INTEGRATION=1 to run the PostgreSQL integration test")

    service = MemoryService(store)
    builder = MemoryContextBuilder(service)

    namespace = "postgres-memory-context-integration"

    try:

        async def exercise() -> None:
            working = await service.remember(
                "Current deployment is using the enterprise AI platform.",
                namespace=namespace,
                memory_type="working",
                metadata={
                    "source": "context-integration-test",
                    "category": "working",
                },
            )

            semantic = await service.remember(
                "The platform uses PostgreSQL for durable conversation memory.",
                namespace=namespace,
                memory_type="semantic",
                metadata={
                    "source": "context-integration-test",
                    "category": "semantic",
                },
            )

            episodic = await service.remember(
                "A previous deployment successfully completed.",
                namespace=namespace,
                memory_type="episodic",
                metadata={
                    "source": "context-integration-test",
                    "category": "episodic",
                },
            )

            context = await builder.build(
                namespace,
                working_limit=5,
                semantic_limit=5,
                episodic_limit=5,
            )

            assert len(context.working) == 1
            assert context.working[0].id == working.id
            assert context.working[0].content == working.content
            assert context.working[0].metadata == working.metadata

            assert len(context.semantic) == 1
            assert context.semantic[0].id == semantic.id
            assert context.semantic[0].content == semantic.content
            assert context.semantic[0].metadata == semantic.metadata

            assert len(context.episodic) == 1
            assert context.episodic[0].id == episodic.id
            assert context.episodic[0].content == episodic.content
            assert context.episodic[0].metadata == episodic.metadata

            assert {item.namespace for item in context.all_items} == {namespace}
            assert len(context.all_items) == 3
            assert not context.is_empty

            await service.forget(working.id)
            await service.forget(semantic.id)
            await service.forget(episodic.id)

        asyncio.run(exercise())

    finally:
        engine.dispose()
