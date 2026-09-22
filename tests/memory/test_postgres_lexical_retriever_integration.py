from __future__ import annotations

import asyncio
import os
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import create_engine, delete, text
from sqlalchemy.orm import Session, sessionmaker

from app.control_plane.persistence.models import MemoryItemRecord
from memory.models import MemoryItem
from memory.retrieval.postgres_lexical import PostgreSQLLexicalMemoryRetriever
from memory.stores.postgres import PostgreSQLMemoryStore


def _postgres_connection():
    if os.getenv("RUN_POSTGRES_INTEGRATION") != "1":
        pytest.skip("Set RUN_POSTGRES_INTEGRATION=1 to run the PostgreSQL integration test")

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

    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
    except Exception:
        engine.dispose()
        raise

    session_factory = sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
    )

    return engine, session_factory


def _item(
    *,
    memory_id: str,
    namespace: str,
    memory_type: str = "episodic",
    content: str,
    created_at: datetime,
    expires_at: datetime | None = None,
    metadata: dict[str, object] | None = None,
) -> MemoryItem:
    return MemoryItem(
        id=memory_id,
        memory_type=memory_type,
        content=content,
        namespace=namespace,
        created_at=created_at,
        expires_at=expires_at,
        metadata=metadata or {},
    )


def _cleanup(
    session_factory,
    memory_ids: list[str],
) -> None:
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


def test_postgresql_lexical_memory_retriever_returns_ranked_matches() -> None:
    engine, session_factory = _postgres_connection()
    store = PostgreSQLMemoryStore(session_factory)
    retriever = PostgreSQLLexicalMemoryRetriever(session_factory=session_factory)

    namespace = "postgres-lexical-memory-ranked"
    now = datetime.now(UTC)
    ids = [
        "postgres-lexical-memory-ranked-exact",
        "postgres-lexical-memory-ranked-partial",
        "postgres-lexical-memory-ranked-unrelated",
    ]

    items = [
        _item(
            memory_id=ids[0],
            namespace=namespace,
            content="Deployment configuration requires approval before production release.",
            created_at=now,
        ),
        _item(
            memory_id=ids[1],
            namespace=namespace,
            content="Production release procedures include deployment configuration checks.",
            created_at=now - timedelta(seconds=1),
        ),
        _item(
            memory_id=ids[2],
            namespace=namespace,
            content="Quarterly finance review completed successfully.",
            created_at=now - timedelta(seconds=2),
        ),
    ]

    try:
        for item in items:
            asyncio.run(store.put(item))

        results = asyncio.run(
            retriever.retrieve(
                "deployment configuration",
                namespace=namespace,
                top_k=2,
            )
        )

        assert [result.item.id for result in results] == [
            ids[0],
            ids[1],
        ]
    finally:
        _cleanup(session_factory, ids)
        engine.dispose()


def test_postgresql_lexical_memory_retriever_returns_older_relevant_memory() -> None:
    engine, session_factory = _postgres_connection()
    store = PostgreSQLMemoryStore(session_factory)
    retriever = PostgreSQLLexicalMemoryRetriever(session_factory=session_factory)

    namespace = "postgres-lexical-memory-older-relevant"
    now = datetime.now(UTC)
    ids = [
        "postgres-lexical-memory-older-relevant-new",
        "postgres-lexical-memory-older-relevant-old",
    ]

    items = [
        _item(
            memory_id=ids[0],
            namespace=namespace,
            content="Unrelated status update for the current project.",
            created_at=now,
        ),
        _item(
            memory_id=ids[1],
            namespace=namespace,
            content="Deployment configuration requires production approval.",
            created_at=now - timedelta(days=1),
        ),
    ]

    try:
        for item in items:
            asyncio.run(store.put(item))

        results = asyncio.run(
            retriever.retrieve(
                "deployment configuration",
                namespace=namespace,
                top_k=1,
            )
        )

        assert [result.item.id for result in results] == [ids[1]]
    finally:
        _cleanup(session_factory, ids)
        engine.dispose()


def test_postgresql_lexical_memory_retriever_applies_namespace_and_type_filters() -> None:
    engine, session_factory = _postgres_connection()
    store = PostgreSQLMemoryStore(session_factory)
    retriever = PostgreSQLLexicalMemoryRetriever(session_factory=session_factory)

    namespace = "postgres-lexical-memory-filter"
    now = datetime.now(UTC)
    ids = [
        "postgres-lexical-memory-filter-episodic",
        "postgres-lexical-memory-filter-semantic",
        "postgres-lexical-memory-filter-other",
    ]

    items = [
        _item(
            memory_id=ids[0],
            namespace=namespace,
            memory_type="episodic",
            content="Deployment configuration was approved.",
            created_at=now,
        ),
        _item(
            memory_id=ids[1],
            namespace=namespace,
            memory_type="semantic",
            content="Deployment configuration requires review.",
            created_at=now - timedelta(seconds=1),
        ),
        _item(
            memory_id=ids[2],
            namespace="different-namespace",
            memory_type="episodic",
            content="Deployment configuration belongs elsewhere.",
            created_at=now - timedelta(seconds=2),
        ),
    ]

    try:
        for item in items:
            asyncio.run(store.put(item))

        episodic = asyncio.run(
            retriever.retrieve(
                "deployment configuration",
                namespace=namespace,
                memory_type="episodic",
                top_k=5,
            )
        )

        assert [result.item.id for result in episodic] == [ids[0]]
    finally:
        _cleanup(session_factory, ids)
        engine.dispose()


def test_postgresql_lexical_memory_retriever_excludes_expired_memories() -> None:
    engine, session_factory = _postgres_connection()
    store = PostgreSQLMemoryStore(session_factory)
    retriever = PostgreSQLLexicalMemoryRetriever(session_factory=session_factory)

    namespace = "postgres-lexical-memory-expiry"
    now = datetime.now(UTC)
    ids = [
        "postgres-lexical-memory-expiry-expired",
        "postgres-lexical-memory-expiry-live",
    ]

    items = [
        _item(
            memory_id=ids[0],
            namespace=namespace,
            content="Deployment configuration was approved.",
            created_at=now,
            expires_at=now - timedelta(seconds=1),
        ),
        _item(
            memory_id=ids[1],
            namespace=namespace,
            content="Deployment configuration requires review.",
            created_at=now - timedelta(seconds=1),
        ),
    ]

    try:
        for item in items:
            asyncio.run(store.put(item))

        results = asyncio.run(
            retriever.retrieve(
                "deployment configuration",
                namespace=namespace,
                top_k=5,
            )
        )

        assert [result.item.id for result in results] == [ids[1]]
    finally:
        _cleanup(session_factory, ids)
        engine.dispose()


def test_postgresql_lexical_memory_retriever_returns_empty_for_no_match() -> None:
    engine, session_factory = _postgres_connection()
    retriever = PostgreSQLLexicalMemoryRetriever(session_factory=session_factory)

    try:
        results = asyncio.run(
            retriever.retrieve(
                "quantum computing",
                namespace="postgres-lexical-memory-no-match",
                top_k=5,
            )
        )

        assert results == ()
    finally:
        engine.dispose()


def test_postgresql_lexical_memory_retriever_validates_arguments() -> None:
    engine, session_factory = _postgres_connection()
    retriever = PostgreSQLLexicalMemoryRetriever(session_factory=session_factory)

    try:
        with pytest.raises(ValueError, match="empty"):
            asyncio.run(
                retriever.retrieve(
                    "   ",
                    namespace="postgres-lexical-memory-validation",
                )
            )

        with pytest.raises(ValueError, match="namespace"):
            asyncio.run(
                retriever.retrieve(
                    "deployment",
                    namespace="   ",
                )
            )

        with pytest.raises(ValueError, match="top_k"):
            asyncio.run(
                retriever.retrieve(
                    "deployment",
                    namespace="postgres-lexical-memory-validation",
                    top_k=0,
                )
            )
    finally:
        engine.dispose()


def test_postgresql_lexical_memory_retriever_preserves_metadata() -> None:
    engine, session_factory = _postgres_connection()
    store = PostgreSQLMemoryStore(session_factory)
    retriever = PostgreSQLLexicalMemoryRetriever(session_factory=session_factory)

    namespace = "postgres-lexical-memory-metadata"
    memory_id = "postgres-lexical-memory-metadata-item"
    now = datetime.now(UTC)

    item = _item(
        memory_id=memory_id,
        namespace=namespace,
        content="Deployment configuration metadata is retained.",
        created_at=now,
        metadata={
            "source": "integration-test",
            "priority": 3,
        },
    )

    try:
        asyncio.run(store.put(item))

        results = asyncio.run(
            retriever.retrieve(
                "deployment configuration",
                namespace=namespace,
                top_k=5,
            )
        )

        assert len(results) == 1
        assert results[0].item.metadata == item.metadata
    finally:
        _cleanup(session_factory, [memory_id])
        engine.dispose()
