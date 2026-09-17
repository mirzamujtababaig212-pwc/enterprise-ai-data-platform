from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine, delete
from sqlalchemy.orm import sessionmaker

from app.control_plane.persistence.models import MemoryItemRecord
from rag.models import EmbeddingIdentity, EmbeddingResult

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_POSTGRES_INTEGRATION") != "1",
    reason="PostgreSQL integration tests require RUN_POSTGRES_INTEGRATION=1",
)


def _session_factory():
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

    with engine.connect() as connection:
        connection.exec_driver_sql("SELECT 1")

    factory = sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
    )

    return engine, factory


def _embedding_result(
    vector: tuple[float, ...] = (1.0, 0.0, 0.0, 0.0),
) -> EmbeddingResult:
    identity = EmbeddingIdentity(
        requested_provider="test-provider",
        requested_model="test-embedding-model",
        resolved_provider="test-provider",
        resolved_model="test-embedding-model",
        dimension=len(vector),
    )

    return EmbeddingResult(
        vector=vector,
        identity=identity,
    )


def _cleanup(
    session_factory,
    memory_id: str,
) -> None:
    session = session_factory()

    try:
        session.execute(
            delete(MemoryItemRecord).where(
                MemoryItemRecord.id == memory_id,
            )
        )
        session.commit()
    finally:
        session.close()


def test_postgresql_memory_embedding_store_round_trips_embedding():
    from memory.embeddings.postgres import PostgreSQLMemoryEmbeddingStore

    engine, session_factory = _session_factory()

    memory_id = "memory-embedding-round-trip"

    try:
        session = session_factory()
        try:
            session.add(
                MemoryItemRecord(
                    id=memory_id,
                    memory_type="semantic",
                    content="Deployment configuration requires production approval.",
                    namespace="memory-embedding-integration",
                    created_at=datetime.now(timezone.utc),
                    expires_at=None,
                    memory_metadata={"source": "integration-test"},
                )
            )
            session.commit()
        finally:
            session.close()

        store = PostgreSQLMemoryEmbeddingStore(session_factory)

        asyncio.run(
            store.put(
                memory_id,
                _embedding_result(),
            )
        )

        result = asyncio.run(store.get(memory_id))

        assert result is not None
        assert result.vector == (1.0, 0.0, 0.0, 0.0)
        assert result.identity.requested_provider == "test-provider"
        assert result.identity.requested_model == "test-embedding-model"
        assert result.identity.resolved_provider == "test-provider"
        assert result.identity.resolved_model == "test-embedding-model"
        assert result.identity.dimension == 4

    finally:
        _cleanup(session_factory, memory_id)
        engine.dispose()


def test_postgresql_memory_embedding_store_replaces_existing_embedding():
    from memory.embeddings.postgres import PostgreSQLMemoryEmbeddingStore

    engine, session_factory = _session_factory()

    memory_id = "memory-embedding-replace"

    try:
        session = session_factory()
        try:
            session.add(
                MemoryItemRecord(
                    id=memory_id,
                    memory_type="semantic",
                    content="Deployment configuration requires production approval.",
                    namespace="memory-embedding-integration",
                    created_at=datetime.now(timezone.utc),
                    expires_at=None,
                    memory_metadata={},
                )
            )
            session.commit()
        finally:
            session.close()

        store = PostgreSQLMemoryEmbeddingStore(session_factory)

        asyncio.run(
            store.put(
                memory_id,
                _embedding_result((1.0, 0.0, 0.0, 0.0)),
            )
        )

        asyncio.run(
            store.put(
                memory_id,
                _embedding_result((0.0, 1.0, 0.0, 0.0)),
            )
        )

        result = asyncio.run(store.get(memory_id))

        assert result is not None
        assert result.vector == (0.0, 1.0, 0.0, 0.0)

    finally:
        _cleanup(session_factory, memory_id)
        engine.dispose()


def test_postgresql_memory_embedding_store_returns_none_when_missing():
    from memory.embeddings.postgres import PostgreSQLMemoryEmbeddingStore

    engine, session_factory = _session_factory()

    try:
        store = PostgreSQLMemoryEmbeddingStore(session_factory)

        result = asyncio.run(store.get("memory-embedding-does-not-exist"))

        assert result is None

    finally:
        engine.dispose()


def test_postgresql_memory_embedding_store_delete():
    from memory.embeddings.postgres import PostgreSQLMemoryEmbeddingStore

    engine, session_factory = _session_factory()

    memory_id = "memory-embedding-delete"

    try:
        session = session_factory()
        try:
            session.add(
                MemoryItemRecord(
                    id=memory_id,
                    memory_type="semantic",
                    content="Deployment configuration requires production approval.",
                    namespace="memory-embedding-integration",
                    created_at=datetime.now(timezone.utc),
                    expires_at=None,
                    memory_metadata={},
                )
            )
            session.commit()
        finally:
            session.close()

        store = PostgreSQLMemoryEmbeddingStore(session_factory)

        asyncio.run(
            store.put(
                memory_id,
                _embedding_result(),
            )
        )

        asyncio.run(store.delete(memory_id))

        assert asyncio.run(store.get(memory_id)) is None

    finally:
        _cleanup(session_factory, memory_id)
        engine.dispose()


def test_postgresql_memory_embedding_store_requires_existing_memory():
    from memory.embeddings.postgres import PostgreSQLMemoryEmbeddingStore

    engine, session_factory = _session_factory()

    try:
        store = PostgreSQLMemoryEmbeddingStore(session_factory)

        with pytest.raises(Exception):
            asyncio.run(
                store.put(
                    "memory-embedding-missing-parent",
                    _embedding_result(),
                )
            )

    finally:
        engine.dispose()
