import pytest

from rag.evaluation.models import RetrievalEvaluationCase


def test_evaluation_case_accepts_binary_relevance() -> None:
    case = RetrievalEvaluationCase(
        query="What is the battery capacity?",
        relevant_chunk_ids=("chunk-17", "chunk-23"),
    )

    assert case.query == "What is the battery capacity?"
    assert case.relevant_chunk_ids == ("chunk-17", "chunk-23")


def test_evaluation_case_accepts_graded_relevance() -> None:
    case = RetrievalEvaluationCase(
        query="What is the battery capacity?",
        relevant_chunk_ids=("chunk-17", "chunk-23"),
        relevance_grades={
            "chunk-17": 3.0,
            "chunk-23": 2.0,
        },
    )

    assert case.relevance_grades == {
        "chunk-17": 3.0,
        "chunk-23": 2.0,
    }


@pytest.mark.parametrize(
    "kwargs",
    [
        {"query": "", "relevant_chunk_ids": ("A",)},
        {"query": "query", "relevant_chunk_ids": ()},
        {"query": "query", "relevant_chunk_ids": ("A", "A")},
        {"query": "query", "relevant_chunk_ids": ("",)},
    ],
)
def test_evaluation_case_rejects_invalid_input(kwargs) -> None:
    with pytest.raises(ValueError):
        RetrievalEvaluationCase(**kwargs)


def test_evaluation_case_rejects_unknown_grade_id() -> None:
    with pytest.raises(ValueError, match="not in relevant_chunk_ids"):
        RetrievalEvaluationCase(
            query="query",
            relevant_chunk_ids=("A",),
            relevance_grades={"B": 2.0},
        )


def test_evaluation_case_rejects_negative_grade() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        RetrievalEvaluationCase(
            query="query",
            relevant_chunk_ids=("A",),
            relevance_grades={"A": -1.0},
        )


def test_evaluation_case_accepts_abstention() -> None:
    case = RetrievalEvaluationCase(
        query="What is a teleportation reactor?",
        relevant_chunk_ids=(),
        expect_abstention=True,
    )

    assert case.expect_abstention is True
    assert case.relevant_chunk_ids == ()


def test_evaluation_case_rejects_abstention_with_relevant_chunks() -> None:
    with pytest.raises(
        ValueError,
        match="expect_abstention cases must not define relevant_chunk_ids",
    ):
        RetrievalEvaluationCase(
            query="unknown topic",
            relevant_chunk_ids=("A",),
            expect_abstention=True,
        )


def test_evaluation_case_rejects_abstention_with_relevance_grades() -> None:
    with pytest.raises(
        ValueError,
        match="expect_abstention cases must not define relevance_grades",
    ):
        RetrievalEvaluationCase(
            query="unknown topic",
            relevant_chunk_ids=(),
            relevance_grades={"A": 1.0},
            expect_abstention=True,
        )
