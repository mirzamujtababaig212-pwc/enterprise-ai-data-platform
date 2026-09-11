import pytest

from rag.evaluation.dataset import RetrievalEvaluationDataset
from rag.evaluation.evaluator import RetrievalEvaluator
from rag.evaluation.models import RetrievalEvaluationCase
from rag.evaluation.policy import RetrievalEvaluationPolicy
from rag.evaluation.workflow import RetrievalEvaluationWorkflow
from rag.models import EmbeddingIdentity


class FakeChunk:
    def __init__(self, chunk_id: str) -> None:
        self.id = chunk_id


class FakeResult:
    def __init__(self, chunk_id: str, score: float = 0.9) -> None:
        self.chunk = FakeChunk(chunk_id)
        self.score = score


TEST_EMBEDDING_IDENTITY = EmbeddingIdentity(
    requested_provider="test-provider",
    requested_model="logical-test-model",
    resolved_provider="test-provider",
    resolved_model="physical-test-model",
    dimension=4,
)


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
    assert result.lineage.dataset_name == "vehicle-retrieval-v1"
    assert result.lineage.dataset_version == "unversioned"
    assert result.lineage.evaluation_policy_name == "vehicle-retrieval-quality-v1"
    assert result.lineage.min_recall_at_k == 1.0
    assert result.lineage.min_precision_at_k == 0.5
    assert result.lineage.min_mrr == 1.0
    assert result.lineage.min_ndcg_at_k == 1.0
    assert result.lineage.max_mean_latency_ms is None
    assert result.lineage.min_abstention_accuracy is None
    assert result.lineage.evaluator_k == 2
    assert result.lineage.min_relevance_score is None
    assert result.lineage.embedding_identity is None
    assert result.evaluation.successful_queries == 1
    assert result.quality_gate.passed is True
    assert result.passed is True


@pytest.mark.asyncio
async def test_workflow_captures_embedding_lineage() -> None:
    dataset = RetrievalEvaluationDataset.from_cases(
        "vehicle-retrieval-v2",
        [
            RetrievalEvaluationCase(
                query="vehicle safety",
                relevant_chunk_ids=("chunk-1",),
            )
        ],
        version="2026-09-11",
    )

    evaluator = RetrievalEvaluator(
        FakeRetriever(),
        k=3,
        min_relevance_score=0.75,
        embedding_identity=TEST_EMBEDDING_IDENTITY,
    )

    workflow = RetrievalEvaluationWorkflow(
        evaluator=evaluator,
        policy=RetrievalEvaluationPolicy(
            name="vehicle-quality-v2",
            min_recall_at_k=1.0,
        ),
    )

    result = await workflow.run(dataset)

    assert result.lineage.dataset_name == "vehicle-retrieval-v2"
    assert result.lineage.dataset_version == "2026-09-11"
    assert result.lineage.evaluation_policy_name == "vehicle-quality-v2"
    assert result.lineage.evaluator_k == 3
    assert result.lineage.min_relevance_score == 0.75
    assert result.lineage.embedding_identity == TEST_EMBEDDING_IDENTITY


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
    assert "lineage" in payload
    assert payload["quality_gate"]["quality_gate_passed"] is True
    assert payload["lineage"]["dataset_name"] == "vehicle-retrieval-v1"
    assert payload["lineage"]["dataset_version"] == "unversioned"
