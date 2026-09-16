from __future__ import annotations

import pytest

from rag.evaluation.datasets.vehicle_retrieval_quality import (
    VEHICLE_QUALITY_BENCHMARK_ITEMS,
    VEHICLE_QUALITY_EVALUATION_CASES,
    VEHICLE_QUALITY_EMBEDDING_IDENTITY,
    VehicleQualityBenchmarkEmbeddingService,
    vehicle_quality_benchmark_chunks,
    vehicle_quality_evaluation_cases,
    vehicle_quality_query_taxonomy,
)


def test_quality_benchmark_has_expected_size_and_unique_ids() -> None:
    assert len(VEHICLE_QUALITY_BENCHMARK_ITEMS) == 24

    chunk_ids = [item.chunk.id for item in VEHICLE_QUALITY_BENCHMARK_ITEMS]

    assert len(chunk_ids) == len(set(chunk_ids))


def test_quality_benchmark_cases_reference_known_chunks() -> None:
    known_ids = {item.chunk.id for item in VEHICLE_QUALITY_BENCHMARK_ITEMS}

    cases = vehicle_quality_evaluation_cases()

    assert cases == VEHICLE_QUALITY_EVALUATION_CASES
    assert len(cases) == 16

    for case in cases:
        assert set(case.relevant_chunk_ids) <= known_ids
        assert case.relevance_grades is not None
        assert set(case.relevance_grades) == set(case.relevant_chunk_ids)


def test_quality_benchmark_chunks_preserve_embedding_identity() -> None:
    chunks = vehicle_quality_benchmark_chunks()

    assert len(chunks) == len(VEHICLE_QUALITY_BENCHMARK_ITEMS)
    assert all(chunk.embedding_identity == VEHICLE_QUALITY_EMBEDDING_IDENTITY for chunk in chunks)


def test_quality_benchmark_chunks_match_canonical_embeddings() -> None:
    chunks = vehicle_quality_benchmark_chunks()

    expected_by_id = {item.chunk.id: item.embedding for item in VEHICLE_QUALITY_BENCHMARK_ITEMS}

    for chunk in chunks:
        assert chunk.embedding == expected_by_id[chunk.chunk.id]
        assert len(chunk.embedding) == VEHICLE_QUALITY_EMBEDDING_IDENTITY.dimension


@pytest.mark.asyncio
async def test_quality_embedding_is_deterministic() -> None:
    service = VehicleQualityBenchmarkEmbeddingService()

    first = await service.embed_with_metadata("regenerative braking battery energy recovery")
    second = await service.embed_with_metadata("regenerative braking battery energy recovery")

    assert first == second
    assert first.identity == VEHICLE_QUALITY_EMBEDDING_IDENTITY


@pytest.mark.asyncio
async def test_quality_embedding_captures_multiple_concepts() -> None:
    service = VehicleQualityBenchmarkEmbeddingService()

    result = await service.embed_with_metadata("hybrid regenerative braking battery")

    assert result.vector[0] == 1.0
    assert result.vector[1] == 1.0
    assert result.vector[2] == 1.0
    assert result.vector[3] == 1.0


def test_quality_benchmark_uses_graded_relevance() -> None:
    for case in vehicle_quality_evaluation_cases():
        assert case.relevance_grades is not None
        assert all(grade >= 0.0 for grade in case.relevance_grades.values())


def test_quality_query_taxonomy_covers_every_case() -> None:
    cases = vehicle_quality_evaluation_cases()
    taxonomy = vehicle_quality_query_taxonomy()

    assert set(taxonomy) == {case.query for case in cases}
    assert set(taxonomy.values()) <= {
        "semantic",
        "lexical",
        "mixed",
        "noisy",
    }


def test_quality_query_taxonomy_has_expected_distribution() -> None:
    taxonomy = vehicle_quality_query_taxonomy()

    counts = {
        category: sum(value == category for value in taxonomy.values())
        for category in {"semantic", "lexical", "mixed", "noisy"}
    }

    assert counts == {
        "semantic": 4,
        "lexical": 9,
        "mixed": 3,
        "noisy": 0,
    }
