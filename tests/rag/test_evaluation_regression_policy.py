import pytest

from rag.evaluation.comparison import (
    RetrievalEvaluationRunComparator,
    RetrievalRegressionPolicy,
    RetrievalRegressionPolicyEvaluator,
)
from tests.rag.test_evaluation_run_comparison import _run


def _comparison(
    *,
    baseline_recall: float = 0.90,
    candidate_recall: float = 0.90,
    baseline_latency: float = 100.0,
    candidate_latency: float = 100.0,
):
    baseline = _run(
        run_id="baseline",
        recall=baseline_recall,
        precision=0.90,
        mrr=0.90,
        ndcg=0.90,
        latency=baseline_latency,
        abstention_accuracy=0.90,
    )

    candidate = _run(
        run_id="candidate",
        recall=candidate_recall,
        precision=0.90,
        mrr=0.90,
        ndcg=0.90,
        latency=candidate_latency,
        abstention_accuracy=0.90,
    )

    return RetrievalEvaluationRunComparator.compare(
        baseline,
        candidate,
    )


def test_policy_allows_degradation_within_tolerance() -> None:
    comparison = _comparison(
        baseline_recall=0.90,
        candidate_recall=0.895,
    )

    policy = RetrievalRegressionPolicy(
        max_recall_at_k_degradation=0.01,
    )

    result = RetrievalRegressionPolicyEvaluator.evaluate(
        comparison,
        policy,
    )

    assert result.passed is True
    assert result.errors == ()


def test_policy_rejects_degradation_beyond_tolerance() -> None:
    comparison = _comparison(
        baseline_recall=0.90,
        candidate_recall=0.88,
    )

    policy = RetrievalRegressionPolicy(
        max_recall_at_k_degradation=0.01,
    )

    result = RetrievalRegressionPolicyEvaluator.evaluate(
        comparison,
        policy,
    )

    assert result.passed is False
    assert len(result.errors) == 1
    assert "recall_at_k" in result.errors[0]


def test_latency_uses_increase_as_degradation() -> None:
    comparison = _comparison(
        baseline_latency=100.0,
        candidate_latency=105.0,
    )

    policy = RetrievalRegressionPolicy(
        max_mean_latency_ms_increase=5.0,
    )

    result = RetrievalRegressionPolicyEvaluator.evaluate(
        comparison,
        policy,
    )

    assert result.passed is True


def test_latency_beyond_tolerance_fails() -> None:
    comparison = _comparison(
        baseline_latency=100.0,
        candidate_latency=106.0,
    )

    policy = RetrievalRegressionPolicy(
        max_mean_latency_ms_increase=5.0,
    )

    result = RetrievalRegressionPolicyEvaluator.evaluate(
        comparison,
        policy,
    )

    assert result.passed is False
    assert any("mean_latency_ms" in error for error in result.errors)


def test_improvements_do_not_fail_policy() -> None:
    comparison = _comparison(
        baseline_recall=0.90,
        candidate_recall=0.95,
        baseline_latency=100.0,
        candidate_latency=90.0,
    )

    policy = RetrievalRegressionPolicy(
        max_recall_at_k_degradation=0.01,
        max_mean_latency_ms_increase=5.0,
    )

    result = RetrievalRegressionPolicyEvaluator.evaluate(
        comparison,
        policy,
    )

    assert result.passed is True
    assert result.errors == ()


@pytest.mark.parametrize(
    "field",
    [
        "max_recall_at_k_degradation",
        "max_precision_at_k_degradation",
        "max_mrr_degradation",
        "max_ndcg_at_k_degradation",
        "max_mean_latency_ms_increase",
        "max_abstention_accuracy_degradation",
    ],
)
def test_policy_rejects_negative_tolerance(field: str) -> None:
    with pytest.raises(ValueError, match=f"{field} must be >= 0"):
        RetrievalRegressionPolicy(**{field: -0.01})


def test_policy_rejects_empty_name() -> None:
    with pytest.raises(ValueError, match="name must not be empty"):
        RetrievalRegressionPolicy(name="   ")


def test_policy_serializes_configuration() -> None:
    policy = RetrievalRegressionPolicy(
        max_recall_at_k_degradation=0.01,
        max_precision_at_k_degradation=0.02,
        max_mrr_degradation=0.01,
        max_ndcg_at_k_degradation=0.01,
        max_mean_latency_ms_increase=10.0,
        max_abstention_accuracy_degradation=0.02,
        name="vehicle-regression-v1",
    )

    payload = policy.as_dict()

    assert payload["name"] == "vehicle-regression-v1"
    assert payload["max_recall_at_k_degradation"] == pytest.approx(0.01)
    assert payload["max_mean_latency_ms_increase"] == pytest.approx(10.0)


def test_policy_result_serializes() -> None:
    comparison = _comparison(
        baseline_recall=0.90,
        candidate_recall=0.88,
    )

    policy = RetrievalRegressionPolicy(
        max_recall_at_k_degradation=0.01,
        name="vehicle-regression-v1",
    )

    result = RetrievalRegressionPolicyEvaluator.evaluate(
        comparison,
        policy,
    )

    payload = result.as_dict()

    assert payload["passed"] is False
    assert payload["policy_name"] == "vehicle-regression-v1"
    assert len(payload["errors"]) == 1
