from datetime import datetime, timezone

import pytest

from memory.evaluation.evaluator import MemoryRetrievalEvaluator
from memory.evaluation.models import MemoryRetrievalEvaluationCase
from memory.models import MemoryItem


def make_item(memory_id: str, content: str) -> MemoryItem:
    return MemoryItem(
        id=memory_id,
        memory_type="semantic",
        content=content,
        namespace="project-a",
        created_at=datetime.now(timezone.utc),
    )


class FakeMemoryRetriever:
    def __init__(self, results_by_query):
        self.results_by_query = results_by_query

    async def retrieve(
        self,
        query,
        *,
        namespace,
        memory_type=None,
        top_k=5,
    ):
        return tuple(self.results_by_query[query][:top_k])


@pytest.mark.asyncio
async def test_memory_retrieval_evaluator_calculates_metrics():
    retriever = FakeMemoryRetriever(
        {
            "deployment": [
                make_item("memory-a", "deployment approval"),
                make_item("memory-x", "unrelated"),
                make_item("memory-b", "deployment configuration"),
            ]
        }
    )

    evaluator = MemoryRetrievalEvaluator(retriever, k=3)

    result = await evaluator.evaluate(
        [
            MemoryRetrievalEvaluationCase(
                query="deployment",
                namespace="project-a",
                relevant_memory_ids=("memory-a", "memory-b"),
            )
        ]
    )

    assert result.evaluated_queries == 1
    assert result.successful_queries == 1
    assert result.failed_queries == 0
    assert result.recall_at_k == pytest.approx(1.0)
    assert result.precision_at_k == pytest.approx(2 / 3)
    assert result.mrr == pytest.approx(1.0)
    assert result.ndcg_at_k > 0.0
    assert result.mean_latency_ms >= 0.0


@pytest.mark.asyncio
async def test_memory_retrieval_evaluator_passes_namespace_and_type():
    class RecordingRetriever:
        def __init__(self):
            self.calls = []

        async def retrieve(
            self,
            query,
            *,
            namespace,
            memory_type=None,
            top_k=5,
        ):
            self.calls.append(
                {
                    "query": query,
                    "namespace": namespace,
                    "memory_type": memory_type,
                    "top_k": top_k,
                }
            )
            return ()

    retriever = RecordingRetriever()
    evaluator = MemoryRetrievalEvaluator(retriever, k=3)

    await evaluator.evaluate(
        [
            MemoryRetrievalEvaluationCase(
                query="deployment",
                namespace="project-a",
                memory_type="episodic",
                relevant_memory_ids=("memory-a",),
            )
        ]
    )

    assert retriever.calls == [
        {
            "query": "deployment",
            "namespace": "project-a",
            "memory_type": "episodic",
            "top_k": 3,
        }
    ]


@pytest.mark.asyncio
async def test_memory_retrieval_evaluator_supports_graded_ndcg():
    retriever = FakeMemoryRetriever(
        {
            "deployment": [
                make_item("memory-b", "deployment"),
                make_item("memory-a", "deployment"),
            ]
        }
    )

    evaluator = MemoryRetrievalEvaluator(retriever, k=2)

    result = await evaluator.evaluate(
        [
            MemoryRetrievalEvaluationCase(
                query="deployment",
                namespace="project-a",
                relevant_memory_ids=("memory-a", "memory-b"),
                relevance_grades={
                    "memory-a": 3.0,
                    "memory-b": 1.0,
                },
            )
        ]
    )

    assert 0.0 < result.ndcg_at_k < 1.0


@pytest.mark.asyncio
async def test_memory_retrieval_evaluator_records_failed_queries():
    class FailingRetriever:
        async def retrieve(
            self,
            query,
            *,
            namespace,
            memory_type=None,
            top_k=5,
        ):
            raise RuntimeError("retrieval failed")

    evaluator = MemoryRetrievalEvaluator(FailingRetriever(), k=3)

    result = await evaluator.evaluate(
        [
            MemoryRetrievalEvaluationCase(
                query="deployment",
                namespace="project-a",
                relevant_memory_ids=("memory-a",),
            )
        ]
    )

    assert result.evaluated_queries == 1
    assert result.successful_queries == 0
    assert result.failed_queries == 1
    assert result.recall_at_k == 0.0
    assert result.precision_at_k == 0.0
    assert result.mrr == 0.0
    assert result.ndcg_at_k == 0.0
    assert result.mean_latency_ms >= 0.0

    query_result = result.query_results[0]
    assert query_result.error_type == "RuntimeError"
    assert query_result.error_message == "retrieval failed"


def test_memory_retrieval_evaluator_rejects_invalid_configuration():
    class Retriever:
        async def retrieve(
            self,
            query,
            *,
            namespace,
            memory_type=None,
            top_k=5,
        ):
            return ()

    with pytest.raises(ValueError, match="k must be greater than zero"):
        MemoryRetrievalEvaluator(Retriever(), k=0)


@pytest.mark.asyncio
async def test_memory_retrieval_evaluator_rejects_empty_cases():
    class Retriever:
        async def retrieve(
            self,
            query,
            *,
            namespace,
            memory_type=None,
            top_k=5,
        ):
            return ()

    evaluator = MemoryRetrievalEvaluator(Retriever(), k=3)

    with pytest.raises(ValueError, match="cases must not be empty"):
        await evaluator.evaluate([])
