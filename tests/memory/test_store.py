from datetime import datetime, timedelta, timezone

import pytest

from memory.models import MemoryItem
from memory.stores.in_memory import InMemoryMemoryStore


@pytest.mark.asyncio
async def test_memory_store_put_get():

    store = InMemoryMemoryStore()

    item = MemoryItem(
        id="memory-1",
        memory_type="semantic",
        content="The project uses Databricks.",
        namespace="project-vehicle-risk",
        created_at=datetime.now(timezone.utc),
    )

    await store.put(item)

    result = await store.get("memory-1")

    assert result == item


@pytest.mark.asyncio
async def test_memory_store_search_filters_namespace():

    store = InMemoryMemoryStore()

    first = MemoryItem(
        id="memory-1",
        memory_type="semantic",
        content="Databricks",
        namespace="project-a",
        created_at=datetime.now(timezone.utc),
    )

    second = MemoryItem(
        id="memory-2",
        memory_type="semantic",
        content="Snowflake",
        namespace="project-b",
        created_at=datetime.now(timezone.utc),
    )

    await store.put(first)
    await store.put(second)

    results = await store.search(
        "project-a",
    )

    assert len(results) == 1
    assert results[0].id == "memory-1"


@pytest.mark.asyncio
async def test_memory_store_filters_memory_type():

    store = InMemoryMemoryStore()

    semantic = MemoryItem(
        id="memory-semantic",
        memory_type="semantic",
        content="Uses Databricks.",
        namespace="project-a",
        created_at=datetime.now(timezone.utc),
    )

    episodic = MemoryItem(
        id="memory-episodic",
        memory_type="episodic",
        content="User discussed Databricks.",
        namespace="project-a",
        created_at=datetime.now(timezone.utc),
    )

    await store.put(semantic)
    await store.put(episodic)

    results = await store.search(
        "project-a",
        memory_type="episodic",
    )

    assert len(results) == 1
    assert results[0].id == "memory-episodic"


@pytest.mark.asyncio
async def test_memory_store_delete():

    store = InMemoryMemoryStore()

    item = MemoryItem(
        id="memory-1",
        memory_type="working",
        content="Temporary context.",
        namespace="session-1",
        created_at=datetime.now(timezone.utc),
    )

    await store.put(item)

    await store.delete("memory-1")

    assert await store.get("memory-1") is None


@pytest.mark.asyncio
async def test_expired_memory_is_not_returned_by_get():
    store = InMemoryMemoryStore()

    item = MemoryItem(
        id="memory-expired",
        memory_type="working",
        content="Temporary context",
        namespace="project-a",
        created_at=datetime.now(timezone.utc),
        expires_at=datetime.now(timezone.utc) - timedelta(minutes=1),
    )

    await store.put(item)

    result = await store.get("memory-expired")

    assert result is None


@pytest.mark.asyncio
async def test_expired_memory_is_not_returned_by_search():
    store = InMemoryMemoryStore()

    expired_item = MemoryItem(
        id="memory-expired",
        memory_type="working",
        content="Expired context",
        namespace="project-a",
        created_at=datetime.now(timezone.utc),
        expires_at=datetime.now(timezone.utc) - timedelta(minutes=1),
    )

    active_item = MemoryItem(
        id="memory-active",
        memory_type="working",
        content="Active context",
        namespace="project-a",
        created_at=datetime.now(timezone.utc),
    )

    await store.put(expired_item)
    await store.put(active_item)

    results = await store.search("project-a")

    assert len(results) == 1
    assert results[0].id == "memory-active"
