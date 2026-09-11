import pytest

from rag.evaluation.dataset import RetrievalEvaluationDataset
from rag.evaluation.evaluator import RetrievalEvaluator
from rag.evaluation.models import RetrievalEvaluationCase
from rag.evaluation.policy import RetrievalEvaluationPolicy
from rag.evaluation.workflow import RetrievalEvaluationWorkflow


class FakeChunk:
    def __init__(self, chunk_id: str) -> None:
        self.id = chunk_id


class FakeResult:
    def __init__(self, chunk_id: str, score: float = 0.9) -> None:
        self.chunk = FakeChunk(chunk_id)
        self.score = score


class FakeRetriever:
    async def retrieve(self, query: str, top_k: int = 5, **kwargs):
        if query == "vehicle safety":
            return [
                FakeResult("chunk-1"),
                FakeResult("chunk-2"),
            ][:top_k]

        return [
            FakeResult("chunk-3"),
            FakeResult("chunk-4"),
        ][:top_k]


@pytest.mark.asyncio
async def test_workflow_runs_evaluation_and_quality_gate() -> None:
    dataset = RetrievalEvaluationDataset.from_cases(
        "vehicle-retrieval-v1",
        [
            RetrievalEvaluationCase(
                query="vehicle safety",
                relevant_chunk_ids=("chunk-1",),
            )
        ],
    )

    evaluator = RetrievalEvaluator(
        FakeRetriever(),
        k=2,
    )

    policy = RetrievalEvaluationPolicy(
        name="vehicle-retrieval-quality-v1",
        min_recall_at_k=1.0,
        min_precision_at_k=0.5,
        min_mrr=1.0,
        min_ndcg_at_k=1.0,
    )

    workflow = RetrievalEvaluationWorkflow(
        evaluator=evaluator,
        policy=policy,
    )

    result = await workflow.run(dataset)

    assert result.dataset_name == "vehicle-retrieval-v1"
    assert result.evaluation.evaluated_queries == 1
    assert result.evaluation.successful_queries == 1
    assert result.quality_gate.passed is True
    assert result.passed is True


@pytest.mark.asyncio
async def test_workflow_returns_failed_quality_gate() -> None:
    dataset = RetrievalEvaluationDataset.from_cases(
        "vehicle-retrieval-v1",
        [
            RetrievalEvaluationCase(
                query="vehicle safety",
                relevant_chunk_ids=("missing-chunk",),
            )
        ],
    )

    evaluator = RetrievalEvaluator(
        FakeRetriever(),
        k=2,
    )

    policy = RetrievalEvaluationPolicy(
        name="strict-retrieval-quality-v1",
        min_recall_at_k=1.0,
    )

    workflow = RetrievalEvaluationWorkflow(
        evaluator=evaluator,
        policy=policy,
    )

    result = await workflow.run(dataset)

    assert result.passed is False
    assert result.quality_gate.passed is False
    assert len(result.quality_gate.errors) == 1


@pytest.mark.asyncio
async def test_workflow_as_dict_contains_evaluation_and_gate() -> None:
    dataset = RetrievalEvaluationDataset.from_cases(
        "vehicle-retrieval-v1",
        [
            RetrievalEvaluationCase(
                query="vehicle safety",
                relevant_chunk_ids=("chunk-1",),
            )
        ],
    )

    workflow = RetrievalEvaluationWorkflow(
        evaluator=RetrievalEvaluator(FakeRetriever(), k=2),
        policy=RetrievalEvaluationPolicy(
            min_recall_at_k=1.0,
        ),
    )

    result = await workflow.run(dataset)
    payload = result.as_dict()

    assert payload["dataset_name"] == "vehicle-retrieval-v1"
    assert "evaluation" in payload
    assert "quality_gate" in payload
    assert payload["quality_gate"]["quality_gate_passed"] is True
