from datetime import datetime, timezone

import pytest

from rag.evaluation.lineage import RetrievalEvaluationLineage
from rag.evaluation.models import (
    RetrievalEvaluationResult,
    RetrievalQueryEvaluation,
)
from rag.evaluation.policy import RetrievalEvaluationPolicy
from rag.evaluation.quality_gate import RetrievalQualityGateResult
from rag.evaluation.run import RetrievalEvaluationRun


def _lineage() -> RetrievalEvaluationLineage:
    return RetrievalEvaluationLineage(
        dataset_name="vehicle-retrieval",
        dataset_version="v2",
        evaluation_policy_name="vehicle-quality-v1",
        min_recall_at_k=0.9,
        min_precision_at_k=0.8,
        min_mrr=0.9,
        min_ndcg_at_k=0.85,
        max_mean_latency_ms=250.0,
        min_abstention_accuracy=0.9,
        evaluator_k=3,
        min_relevance_score=0.75,
    )


def _evaluation() -> RetrievalEvaluationResult:
    return RetrievalEvaluationResult(
        recall_at_k=1.0,
        precision_at_k=1.0,
        mrr=1.0,
        ndcg_at_k=1.0,
        evaluated_queries=1,
        successful_queries=1,
        failed_queries=0,
        mean_latency_ms=10.0,
        query_results=(
            RetrievalQueryEvaluation(
                query="Which vehicle uses an electric powertrain?",
                retrieved_chunk_ids=("vehicle-electric-powertrain",),
                retrieved_results=(),
                relevant_chunk_ids=("vehicle-electric-powertrain",),
                recall_at_k=1.0,
                precision_at_k=1.0,
                reciprocal_rank=1.0,
                ndcg_at_k=1.0,
                latency_ms=10.0,
            ),
        ),
        abstention_accuracy=1.0,
        abstention_evaluated_queries=0,
    )


def _policy() -> RetrievalEvaluationPolicy:
    return RetrievalEvaluationPolicy(
        name="vehicle-quality-v1",
        min_recall_at_k=0.9,
        min_precision_at_k=0.8,
        min_mrr=0.9,
        min_ndcg_at_k=0.85,
        max_mean_latency_ms=250.0,
        min_abstention_accuracy=0.9,
    )


def _quality_gate() -> RetrievalQualityGateResult:
    return RetrievalQualityGateResult(
        passed=True,
        errors=(),
        metrics={
            "recall_at_k": 1.0,
            "precision_at_k": 1.0,
            "mrr": 1.0,
            "ndcg_at_k": 1.0,
            "mean_latency_ms": 10.0,
            "abstention_accuracy": 1.0,
        },
        policy=_policy(),
    )


def _run() -> RetrievalEvaluationRun:
    return RetrievalEvaluationRun(
        run_id="run-vehicle-001",
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        lineage=_lineage(),
        evaluation=_evaluation(),
        quality_gate=_quality_gate(),
    )


def test_run_requires_non_empty_run_id() -> None:
    with pytest.raises(ValueError, match="run_id must not be empty"):
        RetrievalEvaluationRun(
            run_id="",
            created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            lineage=_lineage(),
            evaluation=_evaluation(),
            quality_gate=_quality_gate(),
        )


def test_run_requires_timezone_aware_timestamp() -> None:
    with pytest.raises(ValueError, match="created_at must be timezone-aware"):
        RetrievalEvaluationRun(
            run_id="run-vehicle-001",
            created_at=datetime(2026, 1, 1),
            lineage=_lineage(),
            evaluation=_evaluation(),
            quality_gate=_quality_gate(),
        )


def test_run_exposes_quality_gate_decision() -> None:
    run = _run()

    assert run.passed is True


def test_run_as_dict_contains_artifact_provenance() -> None:
    run = _run()

    payload = run.as_dict()

    assert payload["run_id"] == "run-vehicle-001"
    assert payload["created_at"] == "2026-01-01T00:00:00+00:00"
    assert payload["lineage"]["dataset_name"] == "vehicle-retrieval"
    assert payload["lineage"]["dataset_version"] == "v2"
    assert payload["lineage"]["evaluation_policy_name"] == "vehicle-quality-v1"
    assert payload["lineage"]["min_recall_at_k"] == 0.9
    assert payload["evaluation"]["retrieval_recall_at_k"] == 1.0
    assert payload["quality_gate"]["quality_gate_passed"] is True
