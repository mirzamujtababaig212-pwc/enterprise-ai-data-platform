from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from memory.context.builder import MemoryContext, MemoryContextBuilder
from memory.models import MemoryItem


def make_memory(
    memory_id: str,
    memory_type: str,
    content: str,
) -> MemoryItem:
    return MemoryItem(
        id=memory_id,
        memory_type=memory_type,
        content=content,
        namespace="project-a",
        created_at=datetime.now(timezone.utc),
    )


@pytest.mark.asyncio
async def test_build_returns_memory_grouped_by_type():
    store = MagicMock()
    store.search = AsyncMock(
        side_effect=[
            [
                make_memory(
                    "working-1",
                    "working",
                    "Current task",
                )
            ],
            [
                make_memory(
                    "semantic-1",
                    "semantic",
                    "Project uses Databricks",
                )
            ],
            [
                make_memory(
                    "episodic-1",
                    "episodic",
                    "Previous deployment completed",
                )
            ],
        ]
    )

    from memory.service import MemoryService

    service = MemoryService(store)
    builder = MemoryContextBuilder(service)

    context = await builder.build("project-a")

    assert isinstance(context, MemoryContext)

    assert len(context.working) == 1
    assert context.working[0].id == "working-1"

    assert len(context.semantic) == 1
    assert context.semantic[0].id == "semantic-1"

    assert len(context.episodic) == 1
    assert context.episodic[0].id == "episodic-1"

    assert len(context.all_items) == 3

    assert not context.is_empty


@pytest.mark.asyncio
async def test_build_uses_requested_limits():
    store = MagicMock()
    store.search = AsyncMock(return_value=[])

    from memory.service import MemoryService

    service = MemoryService(store)
    builder = MemoryContextBuilder(service)

    await builder.build(
        "project-a",
        working_limit=3,
        semantic_limit=7,
        episodic_limit=2,
    )

    assert store.search.await_count == 3

    calls = store.search.await_args_list

    assert calls[0].args == ("project-a",)
    assert calls[0].kwargs == {
        "memory_type": "working",
        "limit": 3,
    }

    assert calls[1].args == ("project-a",)
    assert calls[1].kwargs == {
        "memory_type": "semantic",
        "limit": 7,
    }

    assert calls[2].args == ("project-a",)
    assert calls[2].kwargs == {
        "memory_type": "episodic",
        "limit": 2,
    }


@pytest.mark.asyncio
async def test_build_empty_namespace_is_rejected():
    store = MagicMock()

    from memory.service import MemoryService

    service = MemoryService(store)
    builder = MemoryContextBuilder(service)

    with pytest.raises(
        ValueError,
        match="Memory namespace must not be empty",
    ):
        await builder.build("")


@pytest.mark.asyncio
async def test_build_rejects_invalid_limits():
    store = MagicMock()

    from memory.service import MemoryService

    service = MemoryService(store)
    builder = MemoryContextBuilder(service)

    with pytest.raises(
        ValueError,
        match="working_limit must be greater than zero",
    ):
        await builder.build(
            "project-a",
            working_limit=0,
        )

    with pytest.raises(
        ValueError,
        match="semantic_limit must be greater than zero",
    ):
        await builder.build(
            "project-a",
            semantic_limit=0,
        )

    with pytest.raises(
        ValueError,
        match="episodic_limit must be greater than zero",
    ):
        await builder.build(
            "project-a",
            episodic_limit=0,
        )


def test_empty_context_properties():
    context = MemoryContext(
        working=(),
        semantic=(),
        episodic=(),
    )

    assert context.all_items == ()
    assert context.is_empty


@pytest.mark.asyncio
async def test_build_uses_retriever_for_query_aware_memory():
    store = MagicMock()
    store.search = AsyncMock(return_value=[])

    from memory.service import MemoryService

    service = MemoryService(store)

    retriever = MagicMock()
    retriever.retrieve = AsyncMock(
        side_effect=[
            [
                make_memory(
                    "semantic-match",
                    "semantic",
                    "Project uses Databricks",
                )
            ],
            [
                make_memory(
                    "episodic-match",
                    "episodic",
                    "Previous Databricks deployment completed",
                )
            ],
        ]
    )

    builder = MemoryContextBuilder(
        service,
        memory_retriever=retriever,
    )

    context = await builder.build(
        "project-a",
        query="How did we deploy Databricks?",
    )

    assert context.working == ()
    assert [item.id for item in context.semantic] == ["semantic-match"]
    assert [item.id for item in context.episodic] == ["episodic-match"]

    assert retriever.retrieve.await_count == 2

    calls = retriever.retrieve.await_args_list

    assert calls[0].args == ("How did we deploy Databricks?",)
    assert calls[0].kwargs == {
        "namespace": "project-a",
        "memory_type": "semantic",
        "top_k": 5,
    }

    assert calls[1].args == ("How did we deploy Databricks?",)
    assert calls[1].kwargs == {
        "namespace": "project-a",
        "memory_type": "episodic",
        "top_k": 5,
    }

    assert store.search.await_count == 1
    assert store.search.await_args.kwargs == {
        "memory_type": "working",
        "limit": 5,
    }


@pytest.mark.asyncio
async def test_build_without_query_preserves_existing_recall_behavior():
    store = MagicMock()
    store.search = AsyncMock(return_value=[])

    from memory.service import MemoryService

    service = MemoryService(store)

    retriever = MagicMock()
    retriever.retrieve = AsyncMock()

    builder = MemoryContextBuilder(
        service,
        memory_retriever=retriever,
    )

    await builder.build("project-a")

    retriever.retrieve.assert_not_awaited()
    assert store.search.await_count == 3


@pytest.mark.asyncio
async def test_build_rejects_empty_query_before_memory_access():
    store = MagicMock()
    store.search = AsyncMock(return_value=[])

    from memory.service import MemoryService

    service = MemoryService(store)
    builder = MemoryContextBuilder(service)

    with pytest.raises(
        ValueError,
        match="Memory query must not be empty",
    ):
        await builder.build(
            "project-a",
            query="   ",
        )

    store.search.assert_not_awaited()
