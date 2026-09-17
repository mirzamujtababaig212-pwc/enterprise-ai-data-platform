from __future__ import annotations

import asyncio
import os
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine, delete, inspect, text
from sqlalchemy.orm import Session, sessionmaker

from app.control_plane.persistence.models import MemoryItemRecord
from memory.models import MemoryItem
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


def _item(
    *,
    memory_id: str,
    namespace: str,
    memory_type: str = "episodic",
    content: str = "PostgreSQL memory integration test",
    created_at: datetime | None = None,
    expires_at: datetime | None = None,
    metadata: dict[str, object] | None = None,
) -> MemoryItem:
    return MemoryItem(
        id=memory_id,
        memory_type=memory_type,
        content=content,
        namespace=namespace,
        created_at=created_at or datetime.now(timezone.utc),
        expires_at=expires_at,
        metadata=metadata or {},
    )


def test_postgresql_memory_store_round_trip_and_metadata() -> None:
    store, engine = _postgres_store()

    if store is None:
        pytest.skip("Set RUN_POSTGRES_INTEGRATION=1 to run the PostgreSQL integration test")

    memory_id = "postgres-memory-round-trip"
    namespace = "postgres-memory-integration-round-trip"

    item = _item(
        memory_id=memory_id,
        namespace=namespace,
        memory_type="semantic",
        content="Enterprise AI memory persists in PostgreSQL.",
        metadata={
            "source": "integration-test",
            "nested": {"priority": 3, "tags": ["memory", "postgres"]},
        },
    )

    try:
        assert inspect(engine).has_table("memory_items")

        asyncio.run(store.put(item))
        result = asyncio.run(store.get(memory_id))

        assert result is not None
        assert result.id == item.id
        assert result.memory_type == item.memory_type
        assert result.content == item.content
        assert result.namespace == item.namespace
        assert result.metadata == item.metadata
        assert result.expires_at is None
        assert result.created_at == item.created_at
        assert result.created_at.tzinfo is not None
    finally:
        _cleanup(engine, [memory_id])
        engine.dispose()


def test_postgresql_memory_store_replaces_existing_item() -> None:
    store, engine = _postgres_store()

    if store is None:
        pytest.skip("Set RUN_POSTGRES_INTEGRATION=1 to run the PostgreSQL integration test")

    memory_id = "postgres-memory-replacement"
    namespace = "postgres-memory-integration-replacement"

    first = _item(
        memory_id=memory_id,
        namespace=namespace,
        content="Original memory",
        metadata={"version": 1},
    )
    replacement = _item(
        memory_id=memory_id,
        namespace=namespace,
        memory_type="semantic",
        content="Replacement memory",
        metadata={"version": 2},
    )

    try:
        asyncio.run(store.put(first))
        asyncio.run(store.put(replacement))

        result = asyncio.run(store.get(memory_id))

        assert result is not None
        assert result.content == "Replacement memory"
        assert result.memory_type == "semantic"
        assert result.metadata == {"version": 2}
    finally:
        _cleanup(engine, [memory_id])
        engine.dispose()


def test_postgresql_memory_store_filters_namespace_and_type() -> None:
    store, engine = _postgres_store()

    if store is None:
        pytest.skip("Set RUN_POSTGRES_INTEGRATION=1 to run the PostgreSQL integration test")

    namespace = "postgres-memory-integration-filter"
    ids = [
        "postgres-memory-filter-episodic",
        "postgres-memory-filter-semantic",
        "postgres-memory-filter-other",
    ]

    items = [
        _item(
            memory_id=ids[0],
            namespace=namespace,
            memory_type="episodic",
            content="Episodic memory",
        ),
        _item(
            memory_id=ids[1],
            namespace=namespace,
            memory_type="semantic",
            content="Semantic memory",
        ),
        _item(
            memory_id=ids[2],
            namespace="different-namespace",
            memory_type="episodic",
            content="Other namespace",
        ),
    ]

    try:
        for item in items:
            asyncio.run(store.put(item))

        all_results = asyncio.run(store.search(namespace, limit=10))
        assert {item.id for item in all_results} == {ids[0], ids[1]}

        semantic_results = asyncio.run(
            store.search(
                namespace,
                memory_type="semantic",
                limit=10,
            )
        )
        assert [item.id for item in semantic_results] == [ids[1]]
    finally:
        _cleanup(engine, ids)
        engine.dispose()


def test_postgresql_memory_store_excludes_expired_items_before_limit() -> None:
    store, engine = _postgres_store()

    if store is None:
        pytest.skip("Set RUN_POSTGRES_INTEGRATION=1 to run the PostgreSQL integration test")

    namespace = "postgres-memory-integration-expiry"
    now = datetime.now(timezone.utc)
    ids = [
        "postgres-memory-expired-newest",
        "postgres-memory-live-newest",
        "postgres-memory-live-oldest",
    ]

    items = [
        _item(
            memory_id=ids[0],
            namespace=namespace,
            content="Expired newest",
            created_at=now,
            expires_at=now - timedelta(seconds=1),
        ),
        _item(
            memory_id=ids[1],
            namespace=namespace,
            content="Live newest",
            created_at=now - timedelta(seconds=1),
        ),
        _item(
            memory_id=ids[2],
            namespace=namespace,
            content="Live oldest",
            created_at=now - timedelta(seconds=2),
        ),
    ]

    try:
        for item in items:
            asyncio.run(store.put(item))

        results = asyncio.run(store.search(namespace, limit=2))

        assert [item.id for item in results] == [
            ids[1],
            ids[2],
        ]
    finally:
        _cleanup(engine, ids)
        engine.dispose()


def test_postgresql_memory_store_orders_and_limits_deterministically() -> None:
    store, engine = _postgres_store()

    if store is None:
        pytest.skip("Set RUN_POSTGRES_INTEGRATION=1 to run the PostgreSQL integration test")

    namespace = "postgres-memory-integration-order"
    timestamp = datetime.now(timezone.utc)
    ids = [
        "postgres-memory-order-b",
        "postgres-memory-order-a",
        "postgres-memory-order-old",
    ]

    items = [
        _item(
            memory_id=ids[0],
            namespace=namespace,
            content="Same timestamp B",
            created_at=timestamp,
        ),
        _item(
            memory_id=ids[1],
            namespace=namespace,
            content="Same timestamp A",
            created_at=timestamp,
        ),
        _item(
            memory_id=ids[2],
            namespace=namespace,
            content="Older memory",
            created_at=timestamp - timedelta(seconds=1),
        ),
    ]

    try:
        for item in items:
            asyncio.run(store.put(item))

        results = asyncio.run(store.search(namespace, limit=2))

        assert [item.id for item in results] == [
            ids[1],
            ids[0],
        ]
    finally:
        _cleanup(engine, ids)
        engine.dispose()


def test_postgresql_memory_store_delete() -> None:
    store, engine = _postgres_store()

    if store is None:
        pytest.skip("Set RUN_POSTGRES_INTEGRATION=1 to run the PostgreSQL integration test")

    memory_id = "postgres-memory-delete"
    namespace = "postgres-memory-integration-delete"

    try:
        asyncio.run(
            store.put(
                _item(
                    memory_id=memory_id,
                    namespace=namespace,
                )
            )
        )

        assert asyncio.run(store.get(memory_id)) is not None

        asyncio.run(store.delete(memory_id))

        assert asyncio.run(store.get(memory_id)) is None
    finally:
        _cleanup(engine, [memory_id])
        engine.dispose()
