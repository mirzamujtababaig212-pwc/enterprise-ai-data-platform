import pytest

from rag.evaluation.policy import RetrievalEvaluationPolicy


def test_policy_accepts_valid_thresholds() -> None:
    policy = RetrievalEvaluationPolicy(
        name="retrieval-v1",
        min_recall_at_k=0.80,
        min_precision_at_k=0.60,
        min_mrr=0.70,
        min_ndcg_at_k=0.75,
        max_mean_latency_ms=500.0,
    )

    assert policy.name == "retrieval-v1"
    assert policy.as_dict() == {
        "min_recall_at_k": 0.80,
        "min_precision_at_k": 0.60,
        "min_mrr": 0.70,
        "min_ndcg_at_k": 0.75,
        "max_mean_latency_ms": 500.0,
    }


@pytest.mark.parametrize(
    "field",
    [
        "min_recall_at_k",
        "min_precision_at_k",
        "min_mrr",
        "min_ndcg_at_k",
    ],
)
def test_policy_rejects_quality_threshold_outside_range(field) -> None:
    with pytest.raises(ValueError, match="between 0.0 and 1.0"):
        RetrievalEvaluationPolicy(**{field: 1.1})


def test_policy_rejects_negative_latency() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        RetrievalEvaluationPolicy(max_mean_latency_ms=-1.0)


def test_policy_rejects_empty_name() -> None:
    with pytest.raises(ValueError, match="name must not be empty"):
        RetrievalEvaluationPolicy(
            name=" ",
            min_recall_at_k=0.8,
        )


def test_policy_requires_at_least_one_threshold() -> None:
    with pytest.raises(
        ValueError,
        match="at least one evaluation threshold",
    ):
        RetrievalEvaluationPolicy()
