from __future__ import annotations

import pytest

from rag.evaluation.dataset_registry import (
    EvaluationDatasetRegistry,
    VehicleRetrievalEvaluationDatasetDefinition,
)
from rag.evaluation.datasets.vehicle import VEHICLE_EMBEDDING_IDENTITY
from rag.evaluation.lineage import RetrievalEvaluationArtifact


def test_vehicle_dataset_definition_is_versioned() -> None:
    definition = VehicleRetrievalEvaluationDatasetDefinition()

    assert definition.name == "vehicle-retrieval"
    assert definition.version == "v2"


def test_vehicle_dataset_definition_builds_dataset() -> None:
    definition = VehicleRetrievalEvaluationDatasetDefinition()

    dataset = definition.build_dataset()

    assert dataset.name == "vehicle-retrieval"
    assert dataset.version == "v2"
    assert dataset.size == 7


def test_vehicle_dataset_definition_preserves_embedding_identity() -> None:
    definition = VehicleRetrievalEvaluationDatasetDefinition()

    assert definition.build_embedding_identity() == VEHICLE_EMBEDDING_IDENTITY


@pytest.mark.asyncio
async def test_vehicle_dataset_definition_builds_isolated_retriever() -> None:
    definition = VehicleRetrievalEvaluationDatasetDefinition()

    retriever = await definition.build_retriever()

    results = await retriever.retrieve(
        "How does regenerative braking work?",
        top_k=3,
    )

    assert results
    assert len(results) <= 3
    assert results[0].chunk.content
    assert results[0].embedding_identity == VEHICLE_EMBEDDING_IDENTITY


def test_registry_requires_explicit_dataset_and_version() -> None:
    definition = EvaluationDatasetRegistry.get(
        name="vehicle-retrieval",
        version="v2",
    )

    assert definition.name == "vehicle-retrieval"
    assert definition.version == "v2"


def test_registry_rejects_unknown_dataset() -> None:
    with pytest.raises(
        ValueError,
        match="evaluation dataset not found",
    ):
        EvaluationDatasetRegistry.get(
            name="vehicle-retrieval",
            version="does-not-exist",
        )


def test_vehicle_dataset_definition_builds_retrieval_artifact() -> None:
    definition = VehicleRetrievalEvaluationDatasetDefinition()

    artifact = definition.build_retrieval_artifact()

    assert artifact == RetrievalEvaluationArtifact(
        retriever_type="SemanticRetriever",
        vector_store_type="InMemoryVectorStore",
    )
