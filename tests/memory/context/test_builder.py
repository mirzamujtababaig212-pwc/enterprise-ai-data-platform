from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from ai_platform.agents.observability import AgentExecutionEventType
from memory.context.builder import MemoryContext, MemoryContextBuilder
from memory.models import MemoryItem
from memory.retrieval.contracts import MemoryRetrievalResult


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
        created_at=datetime.now(UTC),
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
                MemoryRetrievalResult(
                    item=make_memory(
                        "semantic-match",
                        "semantic",
                        "Project uses Databricks",
                    ),
                    retrieval_method="fake.semantic",
                    rank=1,
                    retrieval_score=0.95,
                )
            ],
            [
                MemoryRetrievalResult(
                    item=make_memory(
                        "episodic-match",
                        "episodic",
                        "Previous Databricks deployment completed",
                    ),
                    retrieval_method="fake.episodic",
                    rank=1,
                    retrieval_score=0.85,
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

    assert len(context.semantic_results) == 1
    assert context.semantic_results[0].item.id == "semantic-match"
    assert context.semantic_results[0].retrieval_method == "fake.semantic"
    assert context.semantic_results[0].retrieval_score == pytest.approx(0.95)

    assert len(context.episodic_results) == 1
    assert context.episodic_results[0].item.id == "episodic-match"
    assert context.episodic_results[0].retrieval_method == "fake.episodic"
    assert context.episodic_results[0].retrieval_score == pytest.approx(0.85)

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


@pytest.mark.asyncio
async def test_build_emits_memory_retrieval_started_and_completed_events():
    store = MagicMock()
    store.search = AsyncMock(return_value=[])

    from memory.service import MemoryService

    service = MemoryService(store)

    retriever = MagicMock()
    retriever.retrieve = AsyncMock(
        side_effect=[
            [
                MemoryRetrievalResult(
                    item=make_memory(
                        "semantic-1",
                        "semantic",
                        "Databricks deployment",
                    ),
                    retrieval_method="hybrid.rrf",
                    rank=1,
                    retrieval_score=0.91,
                    reranker_score=0.88,
                    provenance={
                        "secret_internal_detail": "must-not-leak",
                    },
                )
            ],
            [],
        ]
    )

    observer = MagicMock()
    observer.record = AsyncMock()

    builder = MemoryContextBuilder(
        service,
        memory_retriever=retriever,
        observer=observer,
    )

    await builder.build(
        "project-a",
        query="How did we deploy Databricks?",
        agent_name="test-agent",
        run_id="run-123",
    )

    assert observer.record.await_count == 4

    events = [call.args[0] for call in observer.record.await_args_list]

    assert [event.event_type for event in events] == [
        AgentExecutionEventType.MEMORY_RETRIEVAL_STARTED,
        AgentExecutionEventType.MEMORY_RETRIEVAL_COMPLETED,
        AgentExecutionEventType.MEMORY_RETRIEVAL_STARTED,
        AgentExecutionEventType.MEMORY_RETRIEVAL_COMPLETED,
    ]

    semantic_started = events[0]
    semantic_completed = events[1]

    assert semantic_started.agent_name == "test-agent"
    assert semantic_started.run_id == "run-123"
    assert semantic_started.metadata == {
        "memory_type": "semantic",
        "requested_top_k": 5,
    }

    assert semantic_completed.agent_name == "test-agent"
    assert semantic_completed.run_id == "run-123"
    assert semantic_completed.metadata["memory_type"] == "semantic"
    assert semantic_completed.metadata["requested_top_k"] == 5
    assert semantic_completed.metadata["returned_count"] == 1
    assert semantic_completed.metadata["retrieval_methods"] == ["hybrid.rrf"]
    assert semantic_completed.metadata["retrieval_score_min"] == pytest.approx(0.91)
    assert semantic_completed.metadata["retrieval_score_max"] == pytest.approx(0.91)
    assert semantic_completed.metadata["reranker_score_min"] == pytest.approx(0.88)
    assert semantic_completed.metadata["reranker_score_max"] == pytest.approx(0.88)
    assert semantic_completed.metadata["latency_ms"] >= 0

    for event in events:
        assert "How did we deploy Databricks?" not in str(event.metadata)
        assert "semantic-1" not in str(event.metadata)
        assert "Databricks deployment" not in str(event.metadata)
        assert "secret_internal_detail" not in str(event.metadata)
        assert "project-a" not in str(event.metadata)


@pytest.mark.asyncio
async def test_build_emits_memory_retrieval_failed_event_and_preserves_exception():
    store = MagicMock()
    store.search = AsyncMock(return_value=[])

    from memory.service import MemoryService

    service = MemoryService(store)

    retriever = MagicMock()
    retriever.retrieve = AsyncMock(
        side_effect=RuntimeError("retrieval backend unavailable"),
    )

    observer = MagicMock()
    observer.record = AsyncMock()

    builder = MemoryContextBuilder(
        service,
        memory_retriever=retriever,
        observer=observer,
    )

    with pytest.raises(
        RuntimeError,
        match="retrieval backend unavailable",
    ):
        await builder.build(
            "project-a",
            query="deployment history",
            agent_name="test-agent",
            run_id="run-456",
        )

    assert observer.record.await_count == 2

    events = [call.args[0] for call in observer.record.await_args_list]

    assert [event.event_type for event in events] == [
        AgentExecutionEventType.MEMORY_RETRIEVAL_STARTED,
        AgentExecutionEventType.MEMORY_RETRIEVAL_FAILED,
    ]

    started, failed = events

    assert started.agent_name == "test-agent"
    assert started.run_id == "run-456"
    assert started.metadata == {
        "memory_type": "semantic",
        "requested_top_k": 5,
    }

    assert failed.agent_name == "test-agent"
    assert failed.run_id == "run-456"
    assert failed.metadata["memory_type"] == "semantic"
    assert failed.metadata["requested_top_k"] == 5
    assert failed.metadata["error_type"] == "RuntimeError"
    assert failed.metadata["latency_ms"] >= 0

    assert "retrieval backend unavailable" not in str(failed.metadata)
    assert "deployment history" not in str(failed.metadata)
    assert "project-a" not in str(failed.metadata)


@pytest.mark.asyncio
async def test_build_does_not_emit_retrieval_events_without_agent_identity():
    store = MagicMock()
    store.search = AsyncMock(return_value=[])

    from memory.service import MemoryService

    service = MemoryService(store)

    retriever = MagicMock()
    retriever.retrieve = AsyncMock(
        return_value=[
            MemoryRetrievalResult(
                item=make_memory(
                    "semantic-1",
                    "semantic",
                    "Databricks deployment",
                ),
                retrieval_method="test.fake",
                rank=1,
                retrieval_score=0.9,
            )
        ],
    )

    observer = MagicMock()
    observer.record = AsyncMock()

    builder = MemoryContextBuilder(
        service,
        memory_retriever=retriever,
        observer=observer,
    )

    context = await builder.build(
        "project-a",
        query="deployment",
    )

    assert [item.id for item in context.semantic] == ["semantic-1"]
    observer.record.assert_not_awaited()
