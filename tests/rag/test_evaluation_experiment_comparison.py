from dataclasses import replace
from datetime import datetime, timezone

import pytest

from rag.evaluation.comparison import (
    RetrievalEvaluationExperimentComparator,
    RetrievalEvaluationMetricStatus,
)
from rag.evaluation.lineage import (
    HybridRetrievalConfiguration,
    RetrievalEvaluationArtifact,
    RetrievalEvaluationLineage,
)
from rag.evaluation.models import RetrievalEvaluationResult
from rag.evaluation.policy import RetrievalEvaluationPolicy
from rag.evaluation.quality_gate import RetrievalQualityGateResult
from rag.evaluation.run import RetrievalEvaluationRun
from rag.models import EmbeddingIdentity

EMBEDDING_IDENTITY = EmbeddingIdentity(
    requested_provider="test-provider",
    requested_model="logical-test-model",
    resolved_provider="test-provider",
    resolved_model="physical-test-model",
    dimension=4,
)


def _run(
    run_id: str,
    *,
    recall: float,
    precision: float,
    mrr: float,
    ndcg: float,
    latency: float,
    abstention_accuracy: float,
    retrieval_artifact: RetrievalEvaluationArtifact | None = None,
    dataset_name: str = "vehicle-retrieval-quality",
    dataset_version: str = "v1",
    evaluator_k: int = 5,
    min_relevance_score: float | None = None,
) -> RetrievalEvaluationRun:
    lineage = RetrievalEvaluationLineage(
        dataset_name=dataset_name,
        dataset_version=dataset_version,
        evaluation_policy_name="vehicle-quality-v1",
        min_recall_at_k=0.9,
        min_precision_at_k=0.8,
        min_mrr=0.9,
        min_ndcg_at_k=0.85,
        max_mean_latency_ms=250.0,
        min_abstention_accuracy=0.9,
        evaluator_k=evaluator_k,
        min_relevance_score=min_relevance_score,
        embedding_identity=EMBEDDING_IDENTITY,
        retrieval_artifact=retrieval_artifact,
    )

    evaluation = RetrievalEvaluationResult(
        recall_at_k=recall,
        precision_at_k=precision,
        mrr=mrr,
        ndcg_at_k=ndcg,
        evaluated_queries=10,
        successful_queries=10,
        failed_queries=0,
        mean_latency_ms=latency,
        query_results=(),
        abstention_accuracy=abstention_accuracy,
        abstention_evaluated_queries=2,
    )

    policy = RetrievalEvaluationPolicy(
        name="vehicle-quality-v1",
        min_recall_at_k=0.9,
        min_precision_at_k=0.8,
        min_mrr=0.9,
        min_ndcg_at_k=0.85,
        max_mean_latency_ms=250.0,
        min_abstention_accuracy=0.9,
    )

    quality_gate = RetrievalQualityGateResult(
        passed=True,
        errors=(),
        metrics={},
        policy=policy,
    )

    return RetrievalEvaluationRun(
        run_id=run_id,
        created_at=datetime(2026, 9, 11, tzinfo=timezone.utc),
        lineage=lineage,
        evaluation=evaluation,
        quality_gate=quality_gate,
    )


def _hybrid_artifact(candidate_k: int = 5) -> RetrievalEvaluationArtifact:
    return RetrievalEvaluationArtifact(
        retriever_type="HybridRetriever",
        vector_store_type="InMemoryVectorStore",
        hybrid_configuration=HybridRetrievalConfiguration(
            candidate_k=candidate_k,
            rrf_k=60,
            semantic_weight=1.0,
            lexical_weight=0.5,
        ),
    )


def test_experiment_comparison_allows_different_retrieval_artifacts() -> None:
    baseline = _run(
        "baseline",
        recall=1.0,
        precision=0.2875,
        mrr=0.96875,
        ndcg=0.966778,
        latency=0.277,
        abstention_accuracy=1.0,
        retrieval_artifact=_hybrid_artifact(),
    )
    candidate_artifact = RetrievalEvaluationArtifact(
        retriever_type="RerankingRetriever",
        vector_store_type="InMemoryVectorStore",
    )
    candidate = _run(
        "candidate",
        recall=1.0,
        precision=0.2875,
        mrr=1.0,
        ndcg=0.964524,
        latency=16.806,
        abstention_accuracy=1.0,
        retrieval_artifact=candidate_artifact,
    )

    comparison = RetrievalEvaluationExperimentComparator.compare(
        baseline,
        candidate,
    )

    assert comparison.baseline_run_id == "baseline"
    assert comparison.candidate_run_id == "candidate"
    assert comparison.metrics["mrr"].status is RetrievalEvaluationMetricStatus.IMPROVED
    assert comparison.metrics["mrr"].delta == pytest.approx(0.03125)
    assert comparison.metrics["ndcg_at_k"].status is RetrievalEvaluationMetricStatus.REGRESSED
    assert comparison.metrics["ndcg_at_k"].delta == pytest.approx(-0.002254)
    assert comparison.metrics["mean_latency_ms"].status is RetrievalEvaluationMetricStatus.REGRESSED


@pytest.mark.parametrize(
    ("field", "candidate_change"),
    [
        ("dataset_name", {"dataset_name": "other-retrieval"}),
        ("dataset_version", {"dataset_version": "v2"}),
        ("evaluator_k", {"evaluator_k": 10}),
        ("min_relevance_score", {"min_relevance_score": 0.5}),
    ],
)
def test_experiment_comparison_rejects_incompatible_evaluation_context(
    field: str,
    candidate_change: dict[str, object],
) -> None:
    baseline = _run(
        "baseline",
        recall=0.9,
        precision=0.8,
        mrr=0.9,
        ndcg=0.85,
        latency=100.0,
        abstention_accuracy=0.9,
    )
    candidate = _run(
        "candidate",
        recall=0.9,
        precision=0.8,
        mrr=0.9,
        ndcg=0.85,
        latency=100.0,
        abstention_accuracy=0.9,
    )

    candidate = replace(
        candidate,
        lineage=replace(candidate.lineage, **candidate_change),
    )

    with pytest.raises(
        ValueError,
        match=f"Evaluation experiments are incompatible for {field}",
    ):
        RetrievalEvaluationExperimentComparator.compare(baseline, candidate)


def test_experiment_comparison_rejects_different_embedding_identity() -> None:
    baseline = _run(
        "baseline",
        recall=0.9,
        precision=0.8,
        mrr=0.9,
        ndcg=0.85,
        latency=100.0,
        abstention_accuracy=0.9,
    )
    candidate = _run(
        "candidate",
        recall=0.9,
        precision=0.8,
        mrr=0.9,
        ndcg=0.85,
        latency=100.0,
        abstention_accuracy=0.9,
    )

    candidate_identity = EmbeddingIdentity(
        requested_provider="other-provider",
        requested_model="other-model",
        resolved_provider="other-provider",
        resolved_model="other-model",
        dimension=4,
    )

    candidate = replace(
        candidate,
        lineage=replace(
            candidate.lineage,
            embedding_identity=candidate_identity,
        ),
    )

    with pytest.raises(
        ValueError,
        match="Evaluation experiments are incompatible for embedding_identity",
    ):
        RetrievalEvaluationExperimentComparator.compare(baseline, candidate)


def test_experiment_comparison_does_not_require_retrieval_artifact() -> None:
    baseline = _run(
        "baseline",
        recall=0.9,
        precision=0.8,
        mrr=0.9,
        ndcg=0.85,
        latency=100.0,
        abstention_accuracy=0.9,
        retrieval_artifact=_hybrid_artifact(5),
    )
    candidate = _run(
        "candidate",
        recall=0.95,
        precision=0.85,
        mrr=0.95,
        ndcg=0.9,
        latency=80.0,
        abstention_accuracy=0.95,
        retrieval_artifact=_hybrid_artifact(10),
    )

    comparison = RetrievalEvaluationExperimentComparator.compare(
        baseline,
        candidate,
    )

    assert comparison.metrics["recall_at_k"].delta == pytest.approx(0.05)
    assert comparison.metrics["precision_at_k"].delta == pytest.approx(0.05)
    assert comparison.metrics["mrr"].delta == pytest.approx(0.05)
    assert comparison.metrics["ndcg_at_k"].delta == pytest.approx(0.05)
    assert comparison.metrics["mean_latency_ms"].delta == pytest.approx(-20.0)


def test_experiment_comparison_as_dict_serializes_metrics() -> None:
    baseline = _run(
        "baseline",
        recall=0.9,
        precision=0.8,
        mrr=0.9,
        ndcg=0.85,
        latency=100.0,
        abstention_accuracy=0.9,
    )
    candidate = _run(
        "candidate",
        recall=0.95,
        precision=0.85,
        mrr=0.95,
        ndcg=0.9,
        latency=80.0,
        abstention_accuracy=0.95,
    )

    comparison = RetrievalEvaluationExperimentComparator.compare(
        baseline,
        candidate,
    )

    serialized = comparison.as_dict()

    assert serialized["baseline_run_id"] == "baseline"
    assert serialized["candidate_run_id"] == "candidate"
    assert serialized["metrics"]["mrr"]["status"] == "improved"
    assert serialized["metrics"]["mrr"]["delta"] == pytest.approx(0.05)
