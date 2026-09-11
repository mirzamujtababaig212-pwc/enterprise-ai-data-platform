import pytest

from rag.evaluation.models import RetrievalEvaluationResult
from rag.evaluation.policy import RetrievalEvaluationPolicy
from rag.evaluation.quality_gate import RetrievalQualityGate


def make_result(
    *,
    recall: float = 0.90,
    precision: float = 0.80,
    mrr: float = 0.85,
    ndcg: float = 0.88,
    latency_ms: float = 200.0,
) -> RetrievalEvaluationResult:
    return RetrievalEvaluationResult(
        recall_at_k=recall,
        precision_at_k=precision,
        mrr=mrr,
        ndcg_at_k=ndcg,
        evaluated_queries=10,
        successful_queries=10,
        failed_queries=0,
        mean_latency_ms=latency_ms,
        query_results=(),
    )


def test_quality_gate_passes_when_all_thresholds_are_met() -> None:
    result = make_result()

    policy = RetrievalEvaluationPolicy(
        name="retrieval-v1",
        min_recall_at_k=0.80,
        min_precision_at_k=0.70,
        min_mrr=0.80,
        min_ndcg_at_k=0.85,
        max_mean_latency_ms=500.0,
    )

    gate_result = RetrievalQualityGate.evaluate(result, policy)

    assert gate_result.passed is True
    assert gate_result.errors == ()
    assert gate_result.policy == policy
    assert gate_result.metrics["retrieval_recall_at_k"] == pytest.approx(0.90)


def test_quality_gate_passes_at_exact_threshold() -> None:
    result = make_result(
        recall=0.80,
        precision=0.70,
        mrr=0.75,
        ndcg=0.85,
        latency_ms=500.0,
    )

    policy = RetrievalEvaluationPolicy(
        min_recall_at_k=0.80,
        min_precision_at_k=0.70,
        min_mrr=0.75,
        min_ndcg_at_k=0.85,
        max_mean_latency_ms=500.0,
    )

    gate_result = RetrievalQualityGate.evaluate(result, policy)

    assert gate_result.passed is True
    assert gate_result.errors == ()


def test_quality_gate_reports_multiple_quality_failures() -> None:
    result = make_result(
        recall=0.60,
        precision=0.50,
        mrr=0.40,
        ndcg=0.55,
    )

    policy = RetrievalEvaluationPolicy(
        min_recall_at_k=0.80,
        min_precision_at_k=0.70,
        min_mrr=0.75,
        min_ndcg_at_k=0.85,
    )

    gate_result = RetrievalQualityGate.evaluate(result, policy)

    assert gate_result.passed is False
    assert len(gate_result.errors) == 4
    assert "retrieval_recall_at_k=0.6000" in gate_result.errors[0]
    assert "retrieval_precision_at_k=0.5000" in gate_result.errors[1]
    assert "retrieval_mrr=0.4000" in gate_result.errors[2]
    assert "retrieval_ndcg_at_k=0.5500" in gate_result.errors[3]


def test_quality_gate_reports_latency_failure() -> None:
    result = make_result(latency_ms=501.0)

    policy = RetrievalEvaluationPolicy(
        max_mean_latency_ms=500.0,
    )

    gate_result = RetrievalQualityGate.evaluate(result, policy)

    assert gate_result.passed is False
    assert gate_result.errors == ("retrieval_mean_latency_ms=501.00 exceeds maximum 500.00",)


def test_quality_gate_allows_partial_policy() -> None:
    result = make_result(recall=0.85)

    policy = RetrievalEvaluationPolicy(
        min_recall_at_k=0.80,
    )

    gate_result = RetrievalQualityGate.evaluate(result, policy)

    assert gate_result.passed is True
    assert gate_result.errors == ()


def test_quality_gate_as_dict_contains_decision() -> None:
    result = make_result()

    policy = RetrievalEvaluationPolicy(
        name="retrieval-v1",
        min_recall_at_k=0.80,
    )

    gate_result = RetrievalQualityGate.evaluate(result, policy)
    payload = gate_result.as_dict()

    assert payload["quality_gate_passed"] is True
    assert payload["quality_gate_policy"] == "retrieval-v1"
    assert payload["quality_gate_errors"] == ()
    assert payload["quality_gate_metrics"]["retrieval_recall_at_k"] == pytest.approx(0.90)


def test_quality_gate_passes_abstention_at_exact_threshold() -> None:
    result = RetrievalEvaluationResult(
        recall_at_k=0.90,
        precision_at_k=0.80,
        mrr=0.85,
        ndcg_at_k=0.88,
        evaluated_queries=10,
        successful_queries=10,
        failed_queries=0,
        mean_latency_ms=200.0,
        query_results=(),
        abstention_accuracy=0.90,
        abstention_evaluated_queries=5,
    )

    policy = RetrievalEvaluationPolicy(
        min_abstention_accuracy=0.90,
    )

    gate_result = RetrievalQualityGate.evaluate(result, policy)

    assert gate_result.passed is True
    assert gate_result.errors == ()


def test_quality_gate_reports_abstention_failure() -> None:
    result = RetrievalEvaluationResult(
        recall_at_k=0.90,
        precision_at_k=0.80,
        mrr=0.85,
        ndcg_at_k=0.88,
        evaluated_queries=10,
        successful_queries=10,
        failed_queries=0,
        mean_latency_ms=200.0,
        query_results=(),
        abstention_accuracy=0.80,
        abstention_evaluated_queries=5,
    )

    policy = RetrievalEvaluationPolicy(
        min_abstention_accuracy=0.90,
    )

    gate_result = RetrievalQualityGate.evaluate(result, policy)

    assert gate_result.passed is False
    assert gate_result.errors == (
        "retrieval_abstention_accuracy=0.8000 is below required minimum 0.9000",
    )


def test_quality_gate_rejects_missing_abstention_evidence() -> None:
    result = make_result()

    policy = RetrievalEvaluationPolicy(
        min_abstention_accuracy=0.90,
    )

    gate_result = RetrievalQualityGate.evaluate(result, policy)

    assert gate_result.passed is False
    assert gate_result.errors == (
        "retrieval_abstention_accuracy is unavailable because "
        "no abstention queries were evaluated",
    )


@pytest.mark.parametrize(
    "threshold",
    [-0.1, 1.1],
)
def test_evaluation_policy_rejects_invalid_abstention_threshold(
    threshold: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="min_abstention_accuracy must be between 0.0 and 1.0",
    ):
        RetrievalEvaluationPolicy(
            min_abstention_accuracy=threshold,
        )


def test_evaluation_policy_includes_abstention_threshold() -> None:
    policy = RetrievalEvaluationPolicy(
        name="rag-v2",
        min_abstention_accuracy=0.95,
    )

    assert policy.as_dict() == {
        "min_abstention_accuracy": 0.95,
    }
