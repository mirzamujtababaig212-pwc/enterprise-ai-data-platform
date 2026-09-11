from __future__ import annotations

from datetime import UTC, datetime

from rag.evaluation.lineage import RetrievalEvaluationLineage
from rag.evaluation.models import RetrievalEvaluationResult
from rag.evaluation.policy import RetrievalEvaluationPolicy
from rag.evaluation.quality_gate import RetrievalQualityGate
from rag.evaluation.release import RetrievalEvaluationReleaseGate
from rag.evaluation.run import RetrievalEvaluationRun


def _run(*, passed: bool = True, recall: float | None = None) -> RetrievalEvaluationRun:
    policy = RetrievalEvaluationPolicy(
        min_recall_at_k=0.8,
        name="release-policy",
    )

    evaluation = RetrievalEvaluationResult(
        recall_at_k=(1.0 if passed else 0.5) if recall is None else recall,
        precision_at_k=1.0,
        mrr=1.0,
        ndcg_at_k=1.0,
        evaluated_queries=1,
        successful_queries=1,
        failed_queries=0,
        mean_latency_ms=10.0,
        query_results=(),
        abstention_accuracy=1.0,
        abstention_evaluated_queries=1,
    )

    quality_gate = RetrievalQualityGate.evaluate(
        evaluation,
        policy,
    )

    return RetrievalEvaluationRun(
        run_id="release-run",
        created_at=datetime(2026, 9, 11, 12, 0, tzinfo=UTC),
        lineage=RetrievalEvaluationLineage(
            dataset_name="vehicle-retrieval",
            dataset_version="v2",
            evaluation_policy_name=policy.name,
            min_recall_at_k=policy.min_recall_at_k,
            min_precision_at_k=policy.min_precision_at_k,
            min_mrr=policy.min_mrr,
            min_ndcg_at_k=policy.min_ndcg_at_k,
            max_mean_latency_ms=policy.max_mean_latency_ms,
            min_abstention_accuracy=policy.min_abstention_accuracy,
            evaluator_k=3,
            min_relevance_score=None,
        ),
        evaluation=evaluation,
        quality_gate=quality_gate,
    )


def test_release_gate_passes_when_quality_gate_passes() -> None:
    run = _run(passed=True)

    decision = RetrievalEvaluationReleaseGate.evaluate(run)

    assert decision.run_id == run.run_id
    assert decision.passed is True
    assert decision.errors == ()
    assert run.release_passed is True


def test_release_gate_blocks_when_quality_gate_fails() -> None:
    run = _run(passed=False)

    decision = RetrievalEvaluationReleaseGate.evaluate(run)

    assert decision.run_id == run.run_id
    assert decision.passed is False
    assert decision.errors == ("retrieval quality gate failed",)
    assert run.passed is False
    assert run.release_passed is False


def test_release_gate_blocks_failed_regression() -> None:
    baseline = _run(passed=True, recall=1.0)
    candidate = _run(passed=True, recall=0.9)

    from rag.evaluation.comparison.regression_policy import (
        RetrievalRegressionPolicy,
    )

    candidate = candidate.with_regression(
        baseline=baseline,
        policy=RetrievalRegressionPolicy(
            max_recall_at_k_degradation=0.0,
        ),
    )

    assert candidate.regression is not None
    assert candidate.regression.passed is False

    decision = RetrievalEvaluationReleaseGate.evaluate(candidate)

    assert decision.passed is False
    assert decision.errors == ("retrieval regression policy failed",)


def test_release_decision_serializes() -> None:
    decision = RetrievalEvaluationReleaseGate.evaluate(_run())

    assert decision.as_dict() == {
        "run_id": "release-run",
        "passed": True,
        "errors": (),
    }
