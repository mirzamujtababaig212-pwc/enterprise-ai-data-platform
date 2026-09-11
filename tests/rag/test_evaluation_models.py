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
