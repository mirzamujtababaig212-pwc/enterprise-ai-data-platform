import pytest

from rag.evaluation.metrics import (
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_recall_at_k() -> None:
    assert recall_at_k(
        ("A", "X", "B"),
        ("A", "B", "C"),
        3,
    ) == pytest.approx(2 / 3)


def test_precision_at_k() -> None:
    assert precision_at_k(
        ("A", "X", "B"),
        ("A", "B", "C"),
        3,
    ) == pytest.approx(2 / 3)


def test_reciprocal_rank_returns_first_relevant_rank() -> None:
    assert reciprocal_rank(
        ("X", "B", "A"),
        ("A", "B"),
    ) == pytest.approx(0.5)


def test_reciprocal_rank_returns_zero_without_relevant_result() -> None:
    assert (
        reciprocal_rank(
            ("X", "Y"),
            ("A", "B"),
        )
        == 0.0
    )


def test_ndcg_at_k_perfect_ranking_is_one() -> None:
    assert ndcg_at_k(
        ("A", "B", "C"),
        {"A": 3.0, "B": 2.0, "C": 1.0},
        3,
    ) == pytest.approx(1.0)


def test_ndcg_at_k_rewards_better_ordering() -> None:
    ideal = ndcg_at_k(
        ("A", "B"),
        {"A": 3.0, "B": 1.0},
        2,
    )

    reversed_ranking = ndcg_at_k(
        ("B", "A"),
        {"A": 3.0, "B": 1.0},
        2,
    )

    assert ideal == pytest.approx(1.0)
    assert 0.0 < reversed_ranking < 1.0


@pytest.mark.parametrize(
    "metric",
    [
        recall_at_k,
        precision_at_k,
    ],
)
def test_metrics_reject_non_positive_k(metric) -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        metric(("A",), ("A",), 0)


def test_ndcg_rejects_non_positive_k() -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        ndcg_at_k(("A",), {"A": 1.0}, 0)
