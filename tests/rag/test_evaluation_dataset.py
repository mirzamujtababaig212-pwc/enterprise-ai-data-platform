import pytest

from rag.evaluation.dataset import RetrievalEvaluationDataset
from rag.evaluation.models import RetrievalEvaluationCase


def make_case(query: str = "vehicle safety") -> RetrievalEvaluationCase:
    return RetrievalEvaluationCase(
        query=query,
        relevant_chunk_ids=("chunk-1",),
    )


def test_dataset_accepts_cases() -> None:
    dataset = RetrievalEvaluationDataset.from_cases(
        "vehicle-risk-retrieval-v1",
        [
            make_case(),
            make_case("braking system"),
        ],
    )

    assert dataset.name == "vehicle-risk-retrieval-v1"
    assert dataset.version == "unversioned"
    assert dataset.size == 2
    assert len(dataset.cases) == 2


def test_dataset_is_immutable() -> None:
    dataset = RetrievalEvaluationDataset.from_cases(
        "vehicle-risk-retrieval-v1",
        [make_case()],
    )

    with pytest.raises(AttributeError):
        dataset.name = "changed"


def test_dataset_rejects_empty_name() -> None:
    with pytest.raises(ValueError, match="name must not be empty"):
        RetrievalEvaluationDataset.from_cases(
            "   ",
            [make_case()],
        )


def test_dataset_accepts_explicit_version() -> None:
    dataset = RetrievalEvaluationDataset.from_cases(
        "vehicle-risk-retrieval",
        [make_case()],
        version="2026-09-11",
    )

    assert dataset.version == "2026-09-11"


def test_dataset_rejects_empty_version() -> None:
    with pytest.raises(ValueError, match="version must not be empty"):
        RetrievalEvaluationDataset.from_cases(
            "vehicle-risk-retrieval",
            [make_case()],
            version="   ",
        )


def test_dataset_rejects_empty_cases() -> None:
    with pytest.raises(ValueError, match="cases must not be empty"):
        RetrievalEvaluationDataset.from_cases(
            "vehicle-risk-retrieval-v1",
            [],
        )


def test_dataset_copies_case_sequence() -> None:
    cases = [make_case()]

    dataset = RetrievalEvaluationDataset.from_cases(
        "vehicle-risk-retrieval-v1",
        cases,
    )

    cases.append(make_case("second query"))

    assert dataset.size == 1
