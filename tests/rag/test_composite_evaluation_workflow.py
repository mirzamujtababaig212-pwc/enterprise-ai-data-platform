from __future__ import annotations

from datetime import UTC, datetime

import pytest

from rag.evaluation.comparison.regression_policy import RetrievalRegressionPolicy
from rag.evaluation.composite_workflow import (
    CompositeEvaluationWorkflow,
)
from rag.evaluation.external import (
    ExternalEvaluationMetricPolicy,
    ExternalEvaluationPolicy,
    ExternalEvaluationReleasePolicy,
    ExternalEvaluationResult,
)
from rag.evaluation.lineage import (
    HybridRetrievalConfiguration,
    RerankerConfiguration,
    RetrievalEvaluationArtifact,
)
from rag.evaluation.workflow import RetrievalEvaluationWorkflowResult


def _native_result() -> RetrievalEvaluationWorkflowResult:
    from rag.evaluation.dataset import RetrievalEvaluationDataset
    from rag.evaluation.datasets.vehicle import (
        VEHICLE_EMBEDDING_IDENTITY,
        VehicleBenchmarkEmbeddingService,
        vehicle_benchmark_chunks,
        vehicle_evaluation_cases,
    )
    from rag.evaluation.evaluator import RetrievalEvaluator
    from rag.evaluation.policy import RetrievalEvaluationPolicy
    from rag.evaluation.workflow import RetrievalEvaluationWorkflow
    from rag.stores import InMemoryVectorStore

    async def build() -> RetrievalEvaluationWorkflowResult:
        vector_store = InMemoryVectorStore()
        await vector_store.upsert(vehicle_benchmark_chunks())

        evaluator = RetrievalEvaluator(
            retriever=__import__(
                "rag.retrieval",
                fromlist=["SemanticRetriever"],
            ).SemanticRetriever(
                embedding_service=VehicleBenchmarkEmbeddingService(),
                vector_store=vector_store,
            ),
            k=3,
            embedding_identity=VEHICLE_EMBEDDING_IDENTITY,
        )

        policy = RetrievalEvaluationPolicy(
            name="test-native-policy",
            min_recall_at_k=1.0,
            min_precision_at_k=0.8,
            min_mrr=1.0,
            min_ndcg_at_k=0.95,
        )

        dataset = RetrievalEvaluationDataset.from_cases(
            "vehicle-retrieval",
            vehicle_evaluation_cases(),
            version="v2",
        )

        workflow = RetrievalEvaluationWorkflow(
            evaluator=evaluator,
            policy=policy,
        )

        return await workflow.run(dataset)

    import asyncio

    return asyncio.run(build())


def _external_result(
    *,
    faithfulness: float = 1.0,
    retrieval_artifact: RetrievalEvaluationArtifact | None = None,
) -> ExternalEvaluationResult:
    return ExternalEvaluationResult(
        provider="ragas",
        evaluator="faithfulness",
        metrics={"faithfulness": faithfulness},
        evaluated_samples=1,
        retrieval_artifact=retrieval_artifact,
    )


def _external_policy() -> ExternalEvaluationPolicy:
    return ExternalEvaluationPolicy(
        name="test-external-policy",
        metrics=(
            ExternalEvaluationMetricPolicy(
                provider="ragas",
                evaluator="faithfulness",
                metric_name="faithfulness",
                minimum_value=0.8,
            ),
        ),
    )


def test_composite_workflow_builds_native_only_run() -> None:
    result = _native_result()

    workflow_result = CompositeEvaluationWorkflow.run(
        result=result,
        run_id="test-native-only",
        created_at=datetime.now(UTC),
        external_release_policy=ExternalEvaluationReleasePolicy(
            name="optional-external",
            required=False,
        ),
    )

    assert workflow_result.run.run_id == "test-native-only"
    assert workflow_result.run.external_evaluations == ()
    assert workflow_result.run.external_quality_gate is None
    assert workflow_result.release_decision.passed is True


def test_composite_workflow_attaches_external_evidence_and_quality_gate() -> None:
    result = _native_result()

    retrieval_artifact = RetrievalEvaluationArtifact(
        retriever_type="HybridRetriever",
        vector_store_type="InMemoryVectorStore",
        hybrid_configuration=HybridRetrievalConfiguration(
            candidate_k=5,
            rrf_k=60,
            semantic_weight=1.0,
            lexical_weight=0.5,
        ),
        reranker_configuration=RerankerConfiguration(
            type="cross_encoder",
            model_id="test-reranker",
            onnx_filename="test-reranker.onnx",
            max_length=4096,
            candidate_k=20,
        ),
    )

    external_result = _external_result(
        retrieval_artifact=retrieval_artifact,
    )

    workflow_result = CompositeEvaluationWorkflow.run(
        result=result,
        run_id="test-external-pass",
        created_at=datetime.now(UTC),
        external_release_policy=ExternalEvaluationReleasePolicy(
            name="required-external",
            required=True,
        ),
        external_evaluations=(external_result,),
        external_policy=_external_policy(),
    )

    assert len(workflow_result.run.external_evaluations) == 1

    persisted_external_result = workflow_result.run.external_evaluations[0]

    assert persisted_external_result == external_result
    assert persisted_external_result.retrieval_artifact == retrieval_artifact

    assert workflow_result.run.external_quality_gate is not None
    assert workflow_result.run.external_quality_gate.passed is True
    assert workflow_result.release_decision.passed is True


def test_composite_workflow_fails_required_external_release_without_evidence() -> None:
    result = _native_result()

    workflow_result = CompositeEvaluationWorkflow.run(
        result=result,
        run_id="test-external-missing",
        created_at=datetime.now(UTC),
        external_release_policy=ExternalEvaluationReleasePolicy(
            name="required-external",
            required=True,
        ),
    )

    assert workflow_result.run.passed is True
    assert workflow_result.release_decision.passed is False
    assert workflow_result.release_decision.errors == (
        "external: required external evaluation evidence is missing",
    )


def test_composite_workflow_fails_external_quality_gate() -> None:
    result = _native_result()

    workflow_result = CompositeEvaluationWorkflow.run(
        result=result,
        run_id="test-external-fail",
        created_at=datetime.now(UTC),
        external_release_policy=ExternalEvaluationReleasePolicy(
            name="required-external",
            required=True,
        ),
        external_evaluations=(_external_result(faithfulness=0.5),),
        external_policy=_external_policy(),
    )

    assert workflow_result.run.external_quality_gate is not None
    assert workflow_result.run.external_quality_gate.passed is False
    assert workflow_result.release_decision.passed is False
    assert workflow_result.release_decision.errors


def test_composite_workflow_serializes_complete_result() -> None:
    result = _native_result()

    workflow_result = CompositeEvaluationWorkflow.run(
        result=result,
        run_id="test-serialization",
        created_at=datetime.now(UTC),
        external_release_policy=ExternalEvaluationReleasePolicy(
            name="optional-external",
            required=False,
        ),
    )

    payload = workflow_result.as_dict()

    assert payload["run"]["run_id"] == "test-serialization"
    assert payload["release_decision"]["run_id"] == "test-serialization"


def test_composite_workflow_preserves_explicit_baseline_selection() -> None:
    result = _native_result()

    baseline = result.to_run(
        run_id="baseline-composite-001",
        created_at=datetime(2026, 9, 10, tzinfo=UTC),
    )

    workflow_result = CompositeEvaluationWorkflow.run(
        result=result,
        run_id="candidate-composite-001",
        created_at=datetime(2026, 9, 11, tzinfo=UTC),
        external_release_policy=ExternalEvaluationReleasePolicy(
            name="optional-external",
            required=False,
        ),
        baseline_run_id="baseline-composite-001",
        baselines=[baseline],
        regression_policy=RetrievalRegressionPolicy(
            name="composite-regression-v1",
        ),
    )

    regression = workflow_result.run.regression

    assert regression is not None
    assert regression.baseline_run_id == baseline.run_id
    assert regression.candidate_run_id == workflow_result.run.run_id
    assert regression.passed is True
    assert workflow_result.release_decision.passed is True


def test_composite_workflow_requires_regression_policy_for_explicit_baseline() -> None:
    result = _native_result()

    with pytest.raises(
        ValueError, match="regression_policy must be provided when baseline_run_id is provided"
    ):
        CompositeEvaluationWorkflow.run(
            result=result,
            run_id="test-invalid-baseline",
            created_at=datetime.now(UTC),
            external_release_policy=ExternalEvaluationReleasePolicy(),
            baseline_run_id="baseline",
            baselines=[],
        )
