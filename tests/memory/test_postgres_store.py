from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from app.control_plane.persistence.models import Base
from memory.models import MemoryItem
from memory.stores.postgres import PostgreSQLMemoryStore


@pytest.fixture()
def store():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)

    session_factory = sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
    )

    yield PostgreSQLMemoryStore(session_factory)

    engine.dispose()


def make_item(
    memory_id: str,
    *,
    memory_type: str = "episodic",
    content: str = "memory content",
    namespace: str = "agent-1",
    created_at: datetime | None = None,
    expires_at: datetime | None = None,
    metadata: dict | None = None,
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


@pytest.mark.asyncio
async def test_put_and_get_round_trip(store):
    item = make_item(
        "memory-1",
        content="User prefers concise answers.",
        metadata={"source": "test"},
    )

    await store.put(item)

    assert_memory_equal(await store.get("memory-1"), item)


@pytest.mark.asyncio
async def test_put_replaces_existing_memory(store):
    original = make_item("memory-1", content="original")
    replacement = make_item(
        "memory-1",
        content="replacement",
        metadata={"version": 2},
    )

    await store.put(original)
    await store.put(replacement)

    assert_memory_equal(await store.get("memory-1"), replacement)


@pytest.mark.asyncio
async def test_get_missing_returns_none(store):
    assert await store.get("missing") is None


@pytest.mark.asyncio
async def test_expired_memory_is_not_returned(store):
    item = make_item(
        "memory-expired",
        expires_at=datetime.now(timezone.utc) - timedelta(seconds=1),
    )

    await store.put(item)

    assert await store.get(item.id) is None


@pytest.mark.asyncio
async def test_search_filters_by_namespace(store):
    await store.put(make_item("memory-1", namespace="agent-1"))
    await store.put(make_item("memory-2", namespace="agent-2"))

    results = await store.search("agent-1")

    assert [item.id for item in results] == ["memory-1"]


@pytest.mark.asyncio
async def test_search_filters_by_memory_type(store):
    await store.put(make_item("memory-episodic", memory_type="episodic"))
    await store.put(make_item("memory-semantic", memory_type="semantic"))

    results = await store.search(
        "agent-1",
        memory_type="semantic",
    )

    assert [item.id for item in results] == ["memory-semantic"]


@pytest.mark.asyncio
async def test_search_excludes_expired_before_limit(store):
    now = datetime.now(timezone.utc)

    await store.put(
        make_item(
            "memory-expired",
            created_at=now + timedelta(seconds=2),
            expires_at=now - timedelta(seconds=1),
        )
    )
    await store.put(
        make_item(
            "memory-live-1",
            created_at=now + timedelta(seconds=1),
        )
    )
    await store.put(
        make_item(
            "memory-live-2",
            created_at=now,
        )
    )

    results = await store.search("agent-1", limit=2)

    assert [item.id for item in results] == [
        "memory-live-1",
        "memory-live-2",
    ]


@pytest.mark.asyncio
async def test_search_orders_newest_first(store):
    now = datetime.now(timezone.utc)

    await store.put(
        make_item(
            "memory-old",
            created_at=now,
        )
    )
    await store.put(
        make_item(
            "memory-new",
            created_at=now + timedelta(seconds=1),
        )
    )

    results = await store.search("agent-1")

    assert [item.id for item in results] == [
        "memory-new",
        "memory-old",
    ]


@pytest.mark.asyncio
async def test_search_is_deterministic_for_equal_timestamps(store):
    created_at = datetime.now(timezone.utc)

    await store.put(make_item("memory-b", created_at=created_at))
    await store.put(make_item("memory-a", created_at=created_at))

    results = await store.search("agent-1")

    assert [item.id for item in results] == [
        "memory-a",
        "memory-b",
    ]


@pytest.mark.asyncio
async def test_search_respects_limit(store):
    for index in range(3):
        await store.put(make_item(f"memory-{index}"))

    results = await store.search("agent-1", limit=2)

    assert len(results) == 2


@pytest.mark.asyncio
async def test_delete_removes_memory(store):
    item = make_item("memory-1")

    await store.put(item)
    await store.delete(item.id)

    assert await store.get(item.id) is None


@pytest.mark.asyncio
async def test_delete_missing_memory_is_noop(store):
    await store.delete("missing")


@pytest.mark.asyncio
async def test_search_nonpositive_limit_returns_empty(store):
    await store.put(make_item("memory-1"))

    assert await store.search("agent-1", limit=0) == []
    assert await store.search("agent-1", limit=-1) == []


def assert_memory_equal(actual: MemoryItem, expected: MemoryItem) -> None:
    assert actual.id == expected.id
    assert actual.memory_type == expected.memory_type
    assert actual.content == expected.content
    assert actual.namespace == expected.namespace
    assert actual.metadata == expected.metadata

    assert actual.created_at.replace(tzinfo=timezone.utc) == expected.created_at

    if actual.expires_at is None or expected.expires_at is None:
        assert actual.expires_at == expected.expires_at
    else:
        assert actual.expires_at.replace(tzinfo=timezone.utc) == expected.expires_at
