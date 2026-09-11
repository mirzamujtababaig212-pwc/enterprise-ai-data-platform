from datetime import datetime, timezone

import pytest

from rag.evaluation.comparison import (
    RetrievalEvaluationMetricStatus,
    RetrievalEvaluationRunComparator,
)
from rag.evaluation.lineage import RetrievalEvaluationLineage
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
    dataset_version: str = "v1",
) -> RetrievalEvaluationRun:
    lineage = RetrievalEvaluationLineage(
        dataset_name="vehicle-retrieval",
        dataset_version=dataset_version,
        evaluation_policy_name="vehicle-quality-v1",
        min_recall_at_k=0.9,
        min_precision_at_k=0.8,
        min_mrr=0.9,
        min_ndcg_at_k=0.85,
        max_mean_latency_ms=250.0,
        min_abstention_accuracy=0.9,
        evaluator_k=3,
        min_relevance_score=0.75,
        embedding_identity=EMBEDDING_IDENTITY,
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


def test_comparison_marks_improved_metrics() -> None:
    baseline = _run(
        "baseline",
        recall=0.90,
        precision=0.80,
        mrr=0.90,
        ndcg=0.85,
        latency=100.0,
        abstention_accuracy=0.90,
    )
    candidate = _run(
        "candidate",
        recall=0.95,
        precision=0.85,
        mrr=0.95,
        ndcg=0.90,
        latency=80.0,
        abstention_accuracy=0.95,
    )

    comparison = RetrievalEvaluationRunComparator.compare(
        baseline,
        candidate,
    )

    assert comparison.passed is True
    assert comparison.regressed is False

    assert all(
        metric.status is RetrievalEvaluationMetricStatus.IMPROVED
        for metric in comparison.metrics.values()
    )


def test_comparison_marks_regressed_metrics() -> None:
    baseline = _run(
        "baseline",
        recall=0.95,
        precision=0.90,
        mrr=0.95,
        ndcg=0.90,
        latency=80.0,
        abstention_accuracy=0.95,
    )
    candidate = _run(
        "candidate",
        recall=0.90,
        precision=0.85,
        mrr=0.90,
        ndcg=0.85,
        latency=100.0,
        abstention_accuracy=0.90,
    )

    comparison = RetrievalEvaluationRunComparator.compare(
        baseline,
        candidate,
    )

    assert comparison.passed is False
    assert comparison.regressed is True

    assert all(
        metric.status is RetrievalEvaluationMetricStatus.REGRESSED
        for metric in comparison.metrics.values()
    )


def test_comparison_marks_unchanged_metrics() -> None:
    baseline = _run(
        "baseline",
        recall=0.90,
        precision=0.80,
        mrr=0.90,
        ndcg=0.85,
        latency=100.0,
        abstention_accuracy=0.90,
    )
    candidate = _run(
        "candidate",
        recall=0.90,
        precision=0.80,
        mrr=0.90,
        ndcg=0.85,
        latency=100.0,
        abstention_accuracy=0.90,
    )

    comparison = RetrievalEvaluationRunComparator.compare(
        baseline,
        candidate,
    )

    assert comparison.passed is True

    assert all(
        metric.status is RetrievalEvaluationMetricStatus.UNCHANGED
        for metric in comparison.metrics.values()
    )


@pytest.mark.parametrize(
    ("field", "candidate_change"),
    [
        ("dataset_name", {"dataset_name": "other-retrieval"}),
        ("dataset_version", {"dataset_version": "v2"}),
        ("evaluator_k", {"evaluator_k": 5}),
        ("min_relevance_score", {"min_relevance_score": 0.80}),
    ],
)
def test_comparison_rejects_incompatible_lineage(
    field: str,
    candidate_change: dict[str, object],
) -> None:
    from dataclasses import replace

    baseline = _run(
        "baseline",
        recall=0.90,
        precision=0.80,
        mrr=0.90,
        ndcg=0.85,
        latency=100.0,
        abstention_accuracy=0.90,
    )
    candidate = _run(
        "candidate",
        recall=0.90,
        precision=0.80,
        mrr=0.90,
        ndcg=0.85,
        latency=100.0,
        abstention_accuracy=0.90,
    )

    candidate = replace(
        candidate,
        lineage=replace(candidate.lineage, **candidate_change),
    )

    with pytest.raises(
        ValueError,
        match=f"Evaluation runs are incompatible for {field}",
    ):
        RetrievalEvaluationRunComparator.compare(baseline, candidate)


def test_comparison_rejects_incompatible_embedding_identity() -> None:
    from dataclasses import replace

    baseline = _run(
        "baseline",
        recall=0.90,
        precision=0.80,
        mrr=0.90,
        ndcg=0.85,
        latency=100.0,
        abstention_accuracy=0.90,
    )
    candidate = _run(
        "candidate",
        recall=0.90,
        precision=0.80,
        mrr=0.90,
        ndcg=0.85,
        latency=100.0,
        abstention_accuracy=0.90,
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
        match="Evaluation runs are incompatible for embedding_identity",
    ):
        RetrievalEvaluationRunComparator.compare(baseline, candidate)


def test_comparison_serializes_metric_deltas() -> None:
    baseline = _run(
        "baseline",
        recall=0.90,
        precision=0.80,
        mrr=0.90,
        ndcg=0.85,
        latency=100.0,
        abstention_accuracy=0.90,
    )
    candidate = _run(
        "candidate",
        recall=0.95,
        precision=0.80,
        mrr=0.90,
        ndcg=0.85,
        latency=80.0,
        abstention_accuracy=0.90,
    )

    payload = RetrievalEvaluationRunComparator.compare(
        baseline,
        candidate,
    ).as_dict()

    assert payload["baseline_run_id"] == "baseline"
    assert payload["candidate_run_id"] == "candidate"
    assert payload["passed"] is True
    assert payload["metrics"]["recall_at_k"]["delta"] == pytest.approx(0.05)
    assert payload["metrics"]["recall_at_k"]["status"] == "improved"
    assert payload["metrics"]["mean_latency_ms"]["delta"] == -20.0
    assert payload["metrics"]["mean_latency_ms"]["status"] == "improved"
