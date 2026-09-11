import pytest

from rag.evaluation.datasets.vehicle import (
    VEHICLE_BENCHMARK_ITEMS,
    VEHICLE_EVALUATION_CASES,
    VEHICLE_EMBEDDING_IDENTITY,
    VehicleBenchmarkEmbeddingService,
    vehicle_benchmark_chunks,
    vehicle_evaluation_cases,
)


def test_vehicle_benchmark_has_expected_knowledge_items() -> None:
    assert len(VEHICLE_BENCHMARK_ITEMS) == 6

    chunk_ids = {item.chunk.id for item in VEHICLE_BENCHMARK_ITEMS}

    assert chunk_ids == {
        "vehicle-electric-powertrain",
        "vehicle-gasoline-engine",
        "vehicle-hybrid-powertrain",
        "vehicle-battery-charging",
        "vehicle-regenerative-braking",
        "vehicle-maintenance",
    }


def test_vehicle_benchmark_chunks_preserve_embedding_identity() -> None:
    chunks = vehicle_benchmark_chunks()

    assert len(chunks) == len(VEHICLE_BENCHMARK_ITEMS)

    assert all(chunk.embedding_identity == VEHICLE_EMBEDDING_IDENTITY for chunk in chunks)


def test_vehicle_benchmark_cases_reference_known_chunks() -> None:
    known_ids = {item.chunk.id for item in VEHICLE_BENCHMARK_ITEMS}

    cases = vehicle_evaluation_cases()

    assert cases == VEHICLE_EVALUATION_CASES

    for case in cases:
        assert set(case.relevant_chunk_ids) <= known_ids


async def _embedding_for(
    service: VehicleBenchmarkEmbeddingService,
    text: str,
):
    return await service.embed_with_metadata(text)


@pytest.mark.asyncio
async def test_vehicle_benchmark_embedding_is_deterministic() -> None:
    service = VehicleBenchmarkEmbeddingService()

    first = await _embedding_for(
        service,
        "How is an electric vehicle battery charged?",
    )
    second = await _embedding_for(
        service,
        "How is an electric vehicle battery charged?",
    )

    assert first == second
    assert first.identity == VEHICLE_EMBEDDING_IDENTITY


@pytest.mark.asyncio
async def test_vehicle_benchmark_embedding_distinguishes_topics() -> None:
    service = VehicleBenchmarkEmbeddingService()

    electric = await _embedding_for(
        service,
        "How does an electric vehicle get its power?",
    )
    gasoline = await _embedding_for(
        service,
        "How does a gasoline vehicle generate propulsion?",
    )

    assert electric.vector != gasoline.vector
    assert electric.vector[0] == 1.0
    assert gasoline.vector[1] == 1.0


def test_vehicle_benchmark_uses_graded_relevance() -> None:
    cases = vehicle_evaluation_cases()

    graded_cases = [case for case in cases if case.relevance_grades is not None]

    assert len(graded_cases) == len(cases)

    for case in graded_cases:
        assert case.relevance_grades is not None
        assert all(grade >= 0.0 for grade in case.relevance_grades.values())


def test_vehicle_benchmark_primary_relevance_is_strictly_highest() -> None:
    cases = vehicle_evaluation_cases()

    for case in cases:
        assert case.relevance_grades is not None

        maximum_grade = max(case.relevance_grades.values())

        assert sum(grade == maximum_grade for grade in case.relevance_grades.values()) >= 1
