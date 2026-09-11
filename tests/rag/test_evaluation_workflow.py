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


@pytest.mark.asyncio
async def test_workflow_result_can_create_evaluation_run() -> None:
    from datetime import datetime, timezone

    from rag.evaluation.run import RetrievalEvaluationRun

    dataset = RetrievalEvaluationDataset.from_cases(
        "vehicle-retrieval-v1",
        [
            RetrievalEvaluationCase(
                query="vehicle safety",
                relevant_chunk_ids=("chunk-1",),
            )
        ],
        version="v1",
    )

    workflow = RetrievalEvaluationWorkflow(
        evaluator=RetrievalEvaluator(FakeRetriever(), k=2),
        policy=RetrievalEvaluationPolicy(
            name="vehicle-quality-v1",
            min_recall_at_k=1.0,
        ),
    )

    result = await workflow.run(dataset)

    run = result.to_run(
        run_id="run-vehicle-001",
        created_at=datetime(2026, 9, 11, 12, 0, tzinfo=timezone.utc),
    )

    assert isinstance(run, RetrievalEvaluationRun)
    assert run.run_id == "run-vehicle-001"
    assert run.created_at == datetime(
        2026,
        9,
        11,
        12,
        0,
        tzinfo=timezone.utc,
    )
    assert run.lineage == result.lineage
    assert run.evaluation == result.evaluation
    assert run.quality_gate == result.quality_gate
    assert run.passed is True


@pytest.mark.asyncio
async def test_workflow_result_to_run_does_not_persist() -> None:
    from datetime import datetime, timezone

    from rag.evaluation.stores import InMemoryRetrievalEvaluationRunStore

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

    run = result.to_run(
        run_id="run-vehicle-002",
        created_at=datetime(2026, 9, 11, tzinfo=timezone.utc),
    )

    store = InMemoryRetrievalEvaluationRunStore()

    assert await store.get(run.run_id) is None


@pytest.mark.asyncio
async def test_workflow_result_to_run_attaches_passing_regression() -> None:
    from datetime import datetime, timezone

    from rag.evaluation.comparison import RetrievalRegressionPolicy

    dataset = RetrievalEvaluationDataset.from_cases(
        "vehicle-retrieval-v1",
        [
            RetrievalEvaluationCase(
                query="vehicle safety",
                relevant_chunk_ids=("chunk-1",),
            )
        ],
        version="v1",
    )

    workflow = RetrievalEvaluationWorkflow(
        evaluator=RetrievalEvaluator(FakeRetriever(), k=2),
        policy=RetrievalEvaluationPolicy(
            name="vehicle-quality-v1",
            min_recall_at_k=1.0,
        ),
    )

    result = await workflow.run(dataset)

    baseline = result.to_run(
        run_id="baseline-vehicle-001",
        created_at=datetime(2026, 9, 10, tzinfo=timezone.utc),
    )

    regression_policy = RetrievalRegressionPolicy(
        name="vehicle-regression-v1",
    )

    candidate = result.to_run(
        run_id="candidate-vehicle-001",
        created_at=datetime(2026, 9, 11, tzinfo=timezone.utc),
        baseline=baseline,
        regression_policy=regression_policy,
    )

    assert candidate.regression is not None
    assert candidate.regression.baseline_run_id == baseline.run_id
    assert candidate.regression.candidate_run_id == candidate.run_id
    assert candidate.regression.passed is True
    assert candidate.passed is True
    assert candidate.release_passed is True


@pytest.mark.asyncio
async def test_workflow_result_to_run_regression_failure_blocks_release() -> None:
    from datetime import datetime, timezone

    from rag.evaluation.comparison import RetrievalRegressionPolicy

    class ConfigurableFakeRetriever:
        def __init__(self, chunk_ids: list[str]) -> None:
            self.chunk_ids = chunk_ids

        async def retrieve(
            self,
            query: str,
            top_k: int = 5,
            **kwargs,
        ):
            return [FakeResult(chunk_id) for chunk_id in self.chunk_ids[:top_k]]

    dataset = RetrievalEvaluationDataset.from_cases(
        "vehicle-retrieval-v1",
        [
            RetrievalEvaluationCase(
                query="vehicle safety",
                relevant_chunk_ids=("chunk-1", "chunk-2"),
            )
        ],
        version="v1",
    )

    baseline_workflow = RetrievalEvaluationWorkflow(
        evaluator=RetrievalEvaluator(
            ConfigurableFakeRetriever(["chunk-1", "chunk-2"]),
            k=2,
        ),
        policy=RetrievalEvaluationPolicy(
            name="vehicle-quality-v1",
            min_recall_at_k=0.5,
        ),
    )

    baseline_result = await baseline_workflow.run(dataset)

    baseline = baseline_result.to_run(
        run_id="baseline-vehicle-regression",
        created_at=datetime(2026, 9, 10, tzinfo=timezone.utc),
    )

    candidate_workflow = RetrievalEvaluationWorkflow(
        evaluator=RetrievalEvaluator(
            ConfigurableFakeRetriever(["chunk-1"]),
            k=2,
        ),
        policy=RetrievalEvaluationPolicy(
            name="vehicle-quality-v1",
            min_recall_at_k=0.5,
        ),
    )

    candidate_result = await candidate_workflow.run(dataset)

    regression_policy = RetrievalRegressionPolicy(
        name="vehicle-regression-strict-v1",
        max_recall_at_k_degradation=0.0,
    )

    candidate = candidate_result.to_run(
        run_id="candidate-vehicle-regression",
        created_at=datetime(2026, 9, 11, tzinfo=timezone.utc),
        baseline=baseline,
        regression_policy=regression_policy,
    )

    assert candidate.passed is True
    assert candidate.regression is not None
    assert candidate.regression.passed is False
    assert candidate.regression.result.passed is False
    assert candidate.release_passed is False


@pytest.mark.asyncio
async def test_workflow_result_to_run_rejects_baseline_without_policy() -> None:
    from datetime import datetime, timezone

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

    baseline = result.to_run(
        run_id="baseline-vehicle-002",
        created_at=datetime(2026, 9, 10, tzinfo=timezone.utc),
    )

    with pytest.raises(
        ValueError,
        match="baseline and regression_policy must be provided together",
    ):
        result.to_run(
            run_id="candidate-vehicle-002",
            created_at=datetime(2026, 9, 11, tzinfo=timezone.utc),
            baseline=baseline,
        )


@pytest.mark.asyncio
async def test_workflow_result_to_run_rejects_policy_without_baseline() -> None:
    from datetime import datetime, timezone

    from rag.evaluation.comparison import RetrievalRegressionPolicy

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

    regression_policy = RetrievalRegressionPolicy(
        name="vehicle-regression-v1",
    )

    with pytest.raises(
        ValueError,
        match="baseline and regression_policy must be provided together",
    ):
        result.to_run(
            run_id="candidate-vehicle-003",
            created_at=datetime(2026, 9, 11, tzinfo=timezone.utc),
            regression_policy=regression_policy,
        )


@pytest.mark.asyncio
async def test_workflow_result_to_run_keeps_original_run_without_regression() -> None:
    from datetime import datetime, timezone

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

    baseline = result.to_run(
        run_id="baseline-vehicle-004",
        created_at=datetime(2026, 9, 10, tzinfo=timezone.utc),
    )

    candidate = result.to_run(
        run_id="candidate-vehicle-004",
        created_at=datetime(2026, 9, 11, tzinfo=timezone.utc),
    )

    assert baseline.regression is None
    assert candidate.regression is None
    assert baseline.release_passed is True
    assert candidate.release_passed is True


@pytest.mark.asyncio
async def test_workflow_result_to_run_selects_explicit_baseline_by_id() -> None:
    from datetime import datetime, timezone

    from rag.evaluation.comparison import RetrievalRegressionPolicy

    dataset = RetrievalEvaluationDataset.from_cases(
        "vehicle-retrieval-v1",
        [
            RetrievalEvaluationCase(
                query="vehicle safety",
                relevant_chunk_ids=("chunk-1",),
            )
        ],
        version="v1",
    )

    workflow = RetrievalEvaluationWorkflow(
        evaluator=RetrievalEvaluator(FakeRetriever(), k=2),
        policy=RetrievalEvaluationPolicy(
            name="vehicle-quality-v1",
            min_recall_at_k=1.0,
        ),
    )

    result = await workflow.run(dataset)

    baseline = result.to_run(
        run_id="baseline-explicit-001",
        created_at=datetime(2026, 9, 10, tzinfo=timezone.utc),
    )

    candidate = result.to_run(
        run_id="candidate-explicit-001",
        created_at=datetime(2026, 9, 11, tzinfo=timezone.utc),
        baseline_run_id="baseline-explicit-001",
        baselines=[baseline],
        regression_policy=RetrievalRegressionPolicy(
            name="vehicle-regression-v1",
        ),
    )

    assert candidate.regression is not None
    assert candidate.regression.baseline_run_id == baseline.run_id
    assert candidate.regression.candidate_run_id == candidate.run_id
    assert candidate.regression.passed is True
    assert candidate.release_passed is True


@pytest.mark.asyncio
async def test_workflow_result_to_run_rejects_missing_explicit_baseline() -> None:
    from datetime import datetime, timezone

    from rag.evaluation.comparison import (
        RetrievalEvaluationBaselineSelectionError,
        RetrievalRegressionPolicy,
    )

    dataset = RetrievalEvaluationDataset.from_cases(
        "vehicle-retrieval-v1",
        [
            RetrievalEvaluationCase(
                query="vehicle safety",
                relevant_chunk_ids=("chunk-1",),
            )
        ],
        version="v1",
    )

    workflow = RetrievalEvaluationWorkflow(
        evaluator=RetrievalEvaluator(FakeRetriever(), k=2),
        policy=RetrievalEvaluationPolicy(
            min_recall_at_k=1.0,
        ),
    )

    result = await workflow.run(dataset)

    with pytest.raises(
        RetrievalEvaluationBaselineSelectionError,
        match="baseline run not found",
    ):
        result.to_run(
            run_id="candidate-explicit-002",
            created_at=datetime(2026, 9, 11, tzinfo=timezone.utc),
            baseline_run_id="missing-baseline",
            baselines=[],
            regression_policy=RetrievalRegressionPolicy(
                name="vehicle-regression-v1",
            ),
        )


@pytest.mark.asyncio
async def test_workflow_result_to_run_requires_baselines_for_explicit_baseline_id() -> None:
    from datetime import datetime, timezone

    from rag.evaluation.comparison import RetrievalRegressionPolicy

    dataset = RetrievalEvaluationDataset.from_cases(
        "vehicle-retrieval-v1",
        [
            RetrievalEvaluationCase(
                query="vehicle safety",
                relevant_chunk_ids=("chunk-1",),
            )
        ],
        version="v1",
    )

    workflow = RetrievalEvaluationWorkflow(
        evaluator=RetrievalEvaluator(FakeRetriever(), k=2),
        policy=RetrievalEvaluationPolicy(
            min_recall_at_k=1.0,
        ),
    )

    result = await workflow.run(dataset)

    with pytest.raises(
        ValueError,
        match="baselines must be provided when baseline_run_id is provided",
    ):
        result.to_run(
            run_id="candidate-explicit-003",
            created_at=datetime(2026, 9, 11, tzinfo=timezone.utc),
            baseline_run_id="baseline-explicit-003",
            regression_policy=RetrievalRegressionPolicy(
                name="vehicle-regression-v1",
            ),
        )


@pytest.mark.asyncio
async def test_workflow_result_to_run_requires_regression_policy_for_explicit_baseline_id() -> None:
    from datetime import datetime, timezone

    dataset = RetrievalEvaluationDataset.from_cases(
        "vehicle-retrieval-v1",
        [
            RetrievalEvaluationCase(
                query="vehicle safety",
                relevant_chunk_ids=("chunk-1",),
            )
        ],
        version="v1",
    )

    workflow = RetrievalEvaluationWorkflow(
        evaluator=RetrievalEvaluator(FakeRetriever(), k=2),
        policy=RetrievalEvaluationPolicy(
            min_recall_at_k=1.0,
        ),
    )

    result = await workflow.run(dataset)

    baseline = result.to_run(
        run_id="baseline-explicit-004",
        created_at=datetime(2026, 9, 10, tzinfo=timezone.utc),
    )

    with pytest.raises(
        ValueError,
        match="regression_policy must be provided when baseline_run_id is provided",
    ):
        result.to_run(
            run_id="candidate-explicit-004",
            created_at=datetime(2026, 9, 11, tzinfo=timezone.utc),
            baseline_run_id=baseline.run_id,
            baselines=[baseline],
        )


@pytest.mark.asyncio
async def test_workflow_result_to_run_rejects_baseline_and_baseline_id_together() -> None:
    from datetime import datetime, timezone

    from rag.evaluation.comparison import RetrievalRegressionPolicy

    dataset = RetrievalEvaluationDataset.from_cases(
        "vehicle-retrieval-v1",
        [
            RetrievalEvaluationCase(
                query="vehicle safety",
                relevant_chunk_ids=("chunk-1",),
            )
        ],
        version="v1",
    )

    workflow = RetrievalEvaluationWorkflow(
        evaluator=RetrievalEvaluator(FakeRetriever(), k=2),
        policy=RetrievalEvaluationPolicy(
            min_recall_at_k=1.0,
        ),
    )

    result = await workflow.run(dataset)

    baseline = result.to_run(
        run_id="baseline-explicit-005",
        created_at=datetime(2026, 9, 10, tzinfo=timezone.utc),
    )

    with pytest.raises(
        ValueError,
        match="baseline and baseline_run_id must not be provided together",
    ):
        result.to_run(
            run_id="candidate-explicit-005",
            created_at=datetime(2026, 9, 11, tzinfo=timezone.utc),
            baseline=baseline,
            baseline_run_id=baseline.run_id,
            baselines=[baseline],
            regression_policy=RetrievalRegressionPolicy(
                name="vehicle-regression-v1",
            ),
        )
