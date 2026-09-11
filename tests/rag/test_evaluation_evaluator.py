import pytest

from rag.evaluation import (
    RetrievalEvaluationCase,
    RetrievalEvaluationResult,
    RetrievalEvaluator,
    RetrievalQueryResult,
)
from rag.models import DocumentChunk, RetrievalResult


def make_result(chunk_id: str, score: float = 0.9) -> RetrievalResult:
    return RetrievalResult(
        chunk=DocumentChunk(
            id=chunk_id,
            document_id=f"doc-{chunk_id}",
            content=f"content-{chunk_id}",
        ),
        score=score,
    )


class FakeRetriever:
    def __init__(self, results_by_query):
        self.results_by_query = results_by_query

    async def retrieve(self, query, top_k=5, **kwargs):
        return self.results_by_query[query][:top_k]


class FailingRetriever:
    async def retrieve(self, query, top_k=5, **kwargs):
        raise RuntimeError("retrieval failed")


@pytest.mark.asyncio
async def test_retrieval_evaluator_calculates_aggregate_metrics() -> None:
    retriever = FakeRetriever(
        {
            "battery": [
                make_result("A"),
                make_result("X"),
                make_result("B"),
            ],
        }
    )

    evaluator = RetrievalEvaluator(retriever, k=3)

    result = await evaluator.evaluate(
        [
            RetrievalEvaluationCase(
                query="battery",
                relevant_chunk_ids=("A", "B", "C"),
            )
        ]
    )

    assert isinstance(result, RetrievalEvaluationResult)
    assert result.evaluated_queries == 1
    assert result.successful_queries == 1
    assert result.failed_queries == 0
    assert result.recall_at_k == pytest.approx(2 / 3)
    assert result.precision_at_k == pytest.approx(2 / 3)
    assert result.mrr == pytest.approx(1.0)
    assert result.ndcg_at_k == pytest.approx(
        ndcg_binary_reference(
            ("A", "X", "B"),
            ("A", "B", "C"),
            3,
        )
    )
    assert result.mean_latency_ms >= 0.0


@pytest.mark.asyncio
async def test_retrieval_evaluator_supports_graded_ndcg() -> None:
    retriever = FakeRetriever(
        {
            "query": [
                make_result("B"),
                make_result("A"),
            ],
        }
    )

    evaluator = RetrievalEvaluator(retriever, k=2)

    result = await evaluator.evaluate(
        [
            RetrievalEvaluationCase(
                query="query",
                relevant_chunk_ids=("A", "B"),
                relevance_grades={
                    "A": 3.0,
                    "B": 1.0,
                },
            )
        ]
    )

    assert result.ndcg_at_k < 1.0
    assert result.ndcg_at_k > 0.0


@pytest.mark.asyncio
async def test_retrieval_evaluator_records_failed_queries() -> None:
    evaluator = RetrievalEvaluator(FailingRetriever(), k=3)

    result = await evaluator.evaluate(
        [
            RetrievalEvaluationCase(
                query="query",
                relevant_chunk_ids=("A",),
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
    assert len(result.query_results) == 1
    assert result.query_results[0].error_type == "RuntimeError"
    assert result.query_results[0].error_message == "retrieval failed"


@pytest.mark.asyncio
async def test_retrieval_evaluator_excludes_failed_queries_from_metrics() -> None:
    class MixedRetriever:
        async def retrieve(self, query, top_k=5, **kwargs):
            if query == "failed":
                raise RuntimeError("retrieval failed")

            return [
                make_result("A"),
                make_result("X"),
            ][:top_k]

    evaluator = RetrievalEvaluator(MixedRetriever(), k=2)

    result = await evaluator.evaluate(
        [
            RetrievalEvaluationCase(
                query="successful",
                relevant_chunk_ids=("A",),
            ),
            RetrievalEvaluationCase(
                query="failed",
                relevant_chunk_ids=("A",),
            ),
        ]
    )

    assert result.evaluated_queries == 2
    assert result.successful_queries == 1
    assert result.failed_queries == 1
    assert result.recall_at_k == pytest.approx(1.0)
    assert result.precision_at_k == pytest.approx(0.5)
    assert result.mrr == pytest.approx(1.0)
    assert result.ndcg_at_k == pytest.approx(1.0)
    assert len(result.query_results) == 2
    assert result.query_results[1].error_type == "RuntimeError"


@pytest.mark.asyncio
async def test_retrieval_evaluator_rejects_empty_cases() -> None:
    evaluator = RetrievalEvaluator(FakeRetriever({}), k=3)

    with pytest.raises(ValueError, match="cases must not be empty"):
        await evaluator.evaluate([])


def ndcg_binary_reference(retrieved, relevant, k):
    # Binary relevance reference used only to make the evaluator test
    # independent from the implementation under test.
    import math

    grades = {chunk_id: 1.0 for chunk_id in relevant}

    dcg = sum(
        (2 ** grades.get(chunk_id, 0.0) - 1.0) / math.log2(rank + 1)
        for rank, chunk_id in enumerate(retrieved[:k], start=1)
    )

    ideal = sorted(grades.values(), reverse=True)[:k]

    idcg = sum((2**grade - 1.0) / math.log2(rank + 1) for rank, grade in enumerate(ideal, start=1))

    return dcg / idcg


@pytest.mark.asyncio
async def test_retrieval_evaluator_preserves_retrieval_scores() -> None:
    retriever = FakeRetriever(
        {
            "battery": [
                make_result("A", score=0.91),
                make_result("B", score=0.73),
            ],
        }
    )

    evaluator = RetrievalEvaluator(retriever, k=2)

    result = await evaluator.evaluate(
        [
            RetrievalEvaluationCase(
                query="battery",
                relevant_chunk_ids=("A", "B"),
            )
        ]
    )

    query_result = result.query_results[0]

    assert query_result.retrieved_chunk_ids == ("A", "B")
    assert query_result.retrieved_results == (
        RetrievalQueryResult(chunk_id="A", score=0.91),
        RetrievalQueryResult(chunk_id="B", score=0.73),
    )


@pytest.mark.asyncio
async def test_retrieval_evaluator_passes_min_relevance_score() -> None:
    class RecordingRetriever:
        def __init__(self) -> None:
            self.min_scores = []

        async def retrieve(self, query, top_k=5, **kwargs):
            self.min_scores.append(kwargs.get("min_score"))
            return [make_result("A", score=0.91)]

    retriever = RecordingRetriever()
    evaluator = RetrievalEvaluator(
        retriever,
        k=1,
        min_relevance_score=0.8,
    )

    await evaluator.evaluate(
        [
            RetrievalEvaluationCase(
                query="battery",
                relevant_chunk_ids=("A",),
            )
        ]
    )

    assert retriever.min_scores == [0.8]


def test_retrieval_evaluator_accepts_none_min_relevance_score() -> None:
    evaluator = RetrievalEvaluator(
        FakeRetriever({}),
        min_relevance_score=None,
    )

    assert evaluator.min_relevance_score is None


@pytest.mark.parametrize("threshold", [-0.1, 1.1])
def test_retrieval_evaluator_rejects_invalid_min_relevance_score(
    threshold: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="min_relevance_score must be between 0.0 and 1.0",
    ):
        RetrievalEvaluator(
            FakeRetriever({}),
            min_relevance_score=threshold,
        )
