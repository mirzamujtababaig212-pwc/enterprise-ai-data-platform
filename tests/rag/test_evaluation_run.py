from datetime import datetime, timezone

import pytest

from rag.evaluation.external import (
    ExternalEvaluationMetricPolicy,
    ExternalEvaluationPolicy,
    ExternalEvaluationResult,
)
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


def test_run_without_regression_is_release_passed() -> None:
    run = _run()

    assert run.passed is True
    assert run.release_passed is True
    assert run.regression is None


def test_run_can_attach_passing_regression() -> None:
    from rag.evaluation.comparison import RetrievalRegressionPolicy

    baseline = _run()

    candidate = RetrievalEvaluationRun(
        run_id="run-vehicle-002",
        created_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
        lineage=_lineage(),
        evaluation=_evaluation(),
        quality_gate=_quality_gate(),
    )

    policy = RetrievalRegressionPolicy(
        name="vehicle-regression-v1",
    )

    evaluated_run = candidate.with_regression(
        baseline=baseline,
        policy=policy,
    )

    assert evaluated_run is not candidate
    assert candidate.regression is None

    assert evaluated_run.regression is not None
    assert evaluated_run.regression.baseline_run_id == "run-vehicle-001"
    assert evaluated_run.regression.passed is True

    assert evaluated_run.passed is True
    assert evaluated_run.release_passed is True


def test_run_regression_failure_blocks_release() -> None:
    from rag.evaluation.comparison import RetrievalRegressionPolicy
    from tests.rag.test_evaluation_run_comparison import _run as comparison_run

    baseline = comparison_run(
        "baseline",
        recall=0.95,
        precision=0.90,
        mrr=0.95,
        ndcg=0.90,
        latency=80.0,
        abstention_accuracy=0.95,
    )

    candidate = comparison_run(
        "candidate",
        recall=0.85,
        precision=0.90,
        mrr=0.95,
        ndcg=0.90,
        latency=80.0,
        abstention_accuracy=0.95,
    )

    evaluated_run = candidate.with_regression(
        baseline=baseline,
        policy=RetrievalRegressionPolicy(
            max_recall_at_k_degradation=0.01,
        ),
    )

    assert evaluated_run.passed is True
    assert evaluated_run.regression is not None
    assert evaluated_run.regression.passed is False
    assert evaluated_run.release_passed is False


def test_run_with_regression_serializes_release_decision() -> None:
    from rag.evaluation.comparison import RetrievalRegressionPolicy

    baseline = _run()

    candidate = RetrievalEvaluationRun(
        run_id="run-vehicle-002",
        created_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
        lineage=_lineage(),
        evaluation=_evaluation(),
        quality_gate=_quality_gate(),
    )

    evaluated_run = candidate.with_regression(
        baseline=baseline,
        policy=RetrievalRegressionPolicy(
            name="vehicle-regression-v1",
        ),
    )

    payload = evaluated_run.as_dict()

    assert payload["passed"] is True
    assert payload["release_passed"] is True
    assert payload["regression"] is not None

    regression = payload["regression"]

    assert regression["baseline_run_id"] == "run-vehicle-001"
    assert regression["candidate_run_id"] == "run-vehicle-002"
    assert regression["passed"] is True
    assert regression["policy"]["name"] == "vehicle-regression-v1"
    assert regression["result"]["passed"] is True


def test_regression_requires_matching_candidate_run_id() -> None:
    from rag.evaluation.comparison import (
        RetrievalEvaluationRunComparator,
        RetrievalRegressionPolicy,
        RetrievalRegressionPolicyEvaluator,
    )
    from rag.evaluation.run import RetrievalEvaluationRegression

    baseline = _run()

    candidate = RetrievalEvaluationRun(
        run_id="run-vehicle-002",
        created_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
        lineage=_lineage(),
        evaluation=_evaluation(),
        quality_gate=_quality_gate(),
    )

    comparison = RetrievalEvaluationRunComparator.compare(
        baseline,
        candidate,
    )

    policy = RetrievalRegressionPolicy()

    result = RetrievalRegressionPolicyEvaluator.evaluate(
        comparison,
        policy,
    )

    regression = RetrievalEvaluationRegression(
        baseline_run_id=baseline.run_id,
        candidate_run_id=candidate.run_id,
        comparison=comparison,
        policy=policy,
        result=result,
    )

    with pytest.raises(
        ValueError,
        match="regression comparison candidate_run_id must match run_id",
    ):
        RetrievalEvaluationRun(
            run_id="different-run-id",
            created_at=datetime(2026, 1, 3, tzinfo=timezone.utc),
            lineage=_lineage(),
            evaluation=_evaluation(),
            quality_gate=_quality_gate(),
            regression=regression,
        )


def _external_faithfulness_result() -> ExternalEvaluationResult:
    return ExternalEvaluationResult(
        provider="ragas",
        evaluator="faithfulness",
        metrics={"faithfulness": 0.91},
        evaluated_samples=7,
    )


def test_run_has_no_external_evaluations_by_default() -> None:
    run = _run()

    assert run.external_evaluations == ()


def test_run_can_attach_external_evaluation_immutably() -> None:
    run = _run()
    result = _external_faithfulness_result()

    evaluated_run = run.with_external_evaluation(result)

    assert evaluated_run is not run
    assert run.external_evaluations == ()
    assert evaluated_run.external_evaluations == (result,)

    assert evaluated_run.passed is True
    assert evaluated_run.release_passed is True


def test_run_can_attach_multiple_external_evaluations() -> None:
    run = _run()

    faithfulness = _external_faithfulness_result()

    answer_relevance = ExternalEvaluationResult(
        provider="ragas",
        evaluator="answer_relevance",
        metrics={"answer_relevance": 0.88},
        evaluated_samples=7,
    )

    evaluated_run = run.with_external_evaluation(faithfulness).with_external_evaluation(
        answer_relevance
    )

    assert evaluated_run.external_evaluations == (
        faithfulness,
        answer_relevance,
    )


def test_run_serializes_external_evaluation_evidence() -> None:
    run = _run().with_external_evaluation(
        _external_faithfulness_result(),
    )

    payload = run.as_dict()

    assert payload["passed"] is True
    assert payload["release_passed"] is True

    assert payload["external_evaluations"] == [
        {
            "provider": "ragas",
            "evaluator": "faithfulness",
            "metrics": {"faithfulness": 0.91},
            "evaluated_samples": 7,
            "metadata": {},
        }
    ]


def test_run_has_no_external_quality_gate_by_default() -> None:
    run = _run()

    assert run.external_quality_gate is None


def test_run_can_attach_external_quality_gate_immutably() -> None:
    run = _run().with_external_evaluation(
        _external_faithfulness_result(),
    )

    policy = ExternalEvaluationPolicy(
        name="ragas-v1",
        metrics=(
            ExternalEvaluationMetricPolicy(
                provider="ragas",
                evaluator="faithfulness",
                metric_name="faithfulness",
                minimum_value=0.90,
            ),
        ),
    )

    evaluated_run = run.with_external_quality_gate(policy)

    assert evaluated_run is not run
    assert run.external_quality_gate is None

    assert evaluated_run.external_quality_gate is not None
    assert evaluated_run.external_quality_gate.passed is True
    assert evaluated_run.external_quality_gate.metrics == {
        "ragas/faithfulness/faithfulness": 0.91,
    }

    assert evaluated_run.passed is True
    assert evaluated_run.release_passed is True


def test_external_quality_gate_failure_does_not_change_release_decision() -> None:
    run = _run().with_external_evaluation(
        _external_faithfulness_result(),
    )

    policy = ExternalEvaluationPolicy(
        name="ragas-v1",
        metrics=(
            ExternalEvaluationMetricPolicy(
                provider="ragas",
                evaluator="faithfulness",
                metric_name="faithfulness",
                minimum_value=0.95,
            ),
        ),
    )

    evaluated_run = run.with_external_quality_gate(policy)

    assert evaluated_run.external_quality_gate is not None
    assert evaluated_run.external_quality_gate.passed is False
    assert evaluated_run.external_quality_gate.errors == (
        "external_faithfulness=0.9100 is below required minimum 0.9500",
    )

    # External evaluation evidence is informational until explicitly
    # integrated into the release policy.
    assert evaluated_run.passed is True
    assert evaluated_run.release_passed is True


def test_run_serializes_external_quality_gate() -> None:
    run = _run().with_external_evaluation(
        _external_faithfulness_result(),
    )

    policy = ExternalEvaluationPolicy(
        name="ragas-v1",
        metrics=(
            ExternalEvaluationMetricPolicy(
                provider="ragas",
                evaluator="faithfulness",
                metric_name="faithfulness",
                minimum_value=0.90,
            ),
        ),
    )

    evaluated_run = run.with_external_quality_gate(policy)
    payload = evaluated_run.as_dict()

    assert payload["external_quality_gate"] == {
        "quality_gate_passed": True,
        "quality_gate_policy": "ragas-v1",
        "quality_gate_errors": (),
        "quality_gate_metrics": {
            "ragas/faithfulness/faithfulness": 0.91,
        },
    }
    assert payload["passed"] is True
    assert payload["release_passed"] is True
