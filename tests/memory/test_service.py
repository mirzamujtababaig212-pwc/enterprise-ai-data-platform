from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from memory.models import MemoryItem
from memory.service import MemoryService


@pytest.mark.asyncio
async def test_remember_creates_and_persists_memory():

    store = MagicMock()
    store.put = AsyncMock()

    service = MemoryService(store)

    item = await service.remember(
        "The project uses Databricks.",
        namespace="project-a",
        memory_type="semantic",
        metadata={
            "source": "user",
        },
    )

    assert isinstance(item, MemoryItem)
    assert item.content == "The project uses Databricks."
    assert item.namespace == "project-a"
    assert item.memory_type == "semantic"
    assert item.metadata == {
        "source": "user",
    }
    assert item.id.startswith("memory-")

    store.put.assert_awaited_once_with(item)


@pytest.mark.asyncio
async def test_recall_delegates_to_store():

    store = MagicMock()

    expected = [
        MemoryItem(
            id="memory-1",
            memory_type="semantic",
            content="Databricks",
            namespace="project-a",
            created_at=datetime.now(timezone.utc),
        )
    ]

    store.search = AsyncMock(
        return_value=expected,
    )

    service = MemoryService(store)

    result = await service.recall(
        "project-a",
        memory_type="semantic",
        limit=5,
    )

    assert result == expected

    store.search.assert_awaited_once_with(
        "project-a",
        memory_type="semantic",
        limit=5,
    )


@pytest.mark.asyncio
async def test_forget_delegates_to_store():

    store = MagicMock()
    store.delete = AsyncMock()

    service = MemoryService(store)

    await service.forget(
        "memory-123",
    )

    store.delete.assert_awaited_once_with(
        "memory-123",
    )


@pytest.mark.asyncio
async def test_remember_rejects_empty_content():

    store = MagicMock()
    service = MemoryService(store)

    with pytest.raises(
        ValueError,
        match="Memory content must not be empty",
    ):
        await service.remember(
            "",
            namespace="project-a",
            memory_type="semantic",
        )


@pytest.mark.asyncio
async def test_remember_rejects_empty_namespace():

    store = MagicMock()
    service = MemoryService(store)

    with pytest.raises(
        ValueError,
        match="Memory namespace must not be empty",
    ):
        await service.remember(
            "Databricks",
            namespace="",
            memory_type="semantic",
        )


@pytest.mark.asyncio
async def test_recall_rejects_invalid_limit():

    store = MagicMock()
    service = MemoryService(store)

    with pytest.raises(
        ValueError,
        match="limit must be greater than zero",
    ):
        await service.recall(
            "project-a",
            limit=0,
        )


@pytest.mark.asyncio
async def test_forget_rejects_empty_id():

    store = MagicMock()
    service = MemoryService(store)

    with pytest.raises(
        ValueError,
        match="memory_id must not be empty",
    ):
        await service.forget("")
