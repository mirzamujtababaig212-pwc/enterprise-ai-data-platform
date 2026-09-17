from memory.evaluation.datasets.memory_retrieval_quality import (
    MEMORY_RETRIEVAL_QUALITY_CASES,
    MEMORY_RETRIEVAL_QUALITY_ITEMS,
    PROJECT_A_NAMESPACE,
    PROJECT_B_NAMESPACE,
)
from memory.models import MemoryItem


def test_memory_retrieval_quality_dataset_has_expected_items():
    assert len(MEMORY_RETRIEVAL_QUALITY_ITEMS) == 17
    assert all(isinstance(item, MemoryItem) for item in MEMORY_RETRIEVAL_QUALITY_ITEMS)


def test_memory_retrieval_quality_dataset_has_expected_cases():
    assert len(MEMORY_RETRIEVAL_QUALITY_CASES) == 11
    assert all(case.query.strip() for case in MEMORY_RETRIEVAL_QUALITY_CASES)


def test_memory_retrieval_quality_dataset_uses_unique_memory_ids():
    memory_ids = [item.id for item in MEMORY_RETRIEVAL_QUALITY_ITEMS]
    assert len(memory_ids) == len(set(memory_ids))


def test_memory_retrieval_quality_dataset_ground_truth_ids_exist():
    memory_ids = {item.id for item in MEMORY_RETRIEVAL_QUALITY_ITEMS}

    for case in MEMORY_RETRIEVAL_QUALITY_CASES:
        assert set(case.relevant_memory_ids) <= memory_ids


def test_memory_retrieval_quality_dataset_contains_graded_case():
    graded_cases = [
        case for case in MEMORY_RETRIEVAL_QUALITY_CASES if case.relevance_grades is not None
    ]

    assert len(graded_cases) == 1
    assert graded_cases[0].relevance_grades == {
        "memory-deployment-configuration": 3.0,
        "memory-deployment-monitoring": 1.0,
    }


def test_memory_retrieval_quality_dataset_contains_namespace_isolation_cases():
    project_a_cases = [
        case for case in MEMORY_RETRIEVAL_QUALITY_CASES if case.namespace == PROJECT_A_NAMESPACE
    ]

    assert len(project_a_cases) == 3
    assert all(case.namespace != PROJECT_B_NAMESPACE for case in project_a_cases)


def test_memory_retrieval_quality_dataset_contains_recency_distractor_case():
    case = next(
        case for case in MEMORY_RETRIEVAL_QUALITY_CASES if case.query == "deployment release"
    )

    assert case.relevant_memory_ids == ("memory-release-deployment-relevant",)

    items_by_id = {item.id: item for item in MEMORY_RETRIEVAL_QUALITY_ITEMS}

    relevant = items_by_id["memory-release-deployment-relevant"]
    distractor = items_by_id["memory-release-deployment-distractor"]

    assert relevant.created_at < distractor.created_at


def test_memory_retrieval_quality_dataset_has_expected_case_shapes():
    for case in MEMORY_RETRIEVAL_QUALITY_CASES:
        assert case.namespace
        assert case.relevant_memory_ids
        assert case.memory_type in {None, "working", "semantic", "episodic"}

        if case.relevance_grades is not None:
            assert set(case.relevance_grades) <= set(case.relevant_memory_ids)
