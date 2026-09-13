from __future__ import annotations

from dataclasses import dataclass

from rag.evaluation.models import RetrievalEvaluationCase
from rag.models import (
    DocumentChunk,
    EmbeddedChunk,
    EmbeddingIdentity,
    EmbeddingResult,
)

VEHICLE_EMBEDDING_IDENTITY = EmbeddingIdentity(
    requested_provider="test-provider",
    requested_model="vehicle-benchmark",
    resolved_provider="test-provider",
    resolved_model="vehicle-benchmark",
    dimension=6,
)


@dataclass(frozen=True)
class VehicleBenchmarkItem:
    chunk: DocumentChunk
    embedding: tuple[float, ...]


VEHICLE_BENCHMARK_ITEMS = (
    VehicleBenchmarkItem(
        chunk=DocumentChunk(
            id="vehicle-electric-powertrain",
            document_id="vehicle-knowledge",
            content=(
                "Electric vehicles use battery-powered electric motors " "to provide propulsion."
            ),
        ),
        embedding=(1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    ),
    VehicleBenchmarkItem(
        chunk=DocumentChunk(
            id="vehicle-gasoline-engine",
            document_id="vehicle-knowledge",
            content=(
                "Gasoline vehicles use internal combustion engines " "to generate propulsion."
            ),
        ),
        embedding=(0.0, 1.0, 0.0, 0.0, 0.0, 0.0),
    ),
    VehicleBenchmarkItem(
        chunk=DocumentChunk(
            id="vehicle-hybrid-powertrain",
            document_id="vehicle-knowledge",
            content=(
                "Hybrid vehicles combine an internal combustion engine " "with an electric motor."
            ),
        ),
        embedding=(0.0, 1.0, 1.0, 0.0, 0.0, 0.0),
    ),
    VehicleBenchmarkItem(
        chunk=DocumentChunk(
            id="vehicle-battery-charging",
            document_id="vehicle-knowledge",
            content=(
                "Electric vehicle batteries can be charged using " "dedicated charging equipment."
            ),
        ),
        embedding=(1.0, 0.0, 0.0, 1.0, 0.0, 0.0),
    ),
    VehicleBenchmarkItem(
        chunk=DocumentChunk(
            id="vehicle-regenerative-braking",
            document_id="vehicle-knowledge",
            content=(
                "Regenerative braking recovers kinetic energy and "
                "stores it in the vehicle battery."
            ),
        ),
        embedding=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
    ),
    VehicleBenchmarkItem(
        chunk=DocumentChunk(
            id="vehicle-maintenance",
            document_id="vehicle-knowledge",
            content=(
                "Routine vehicle maintenance includes inspections, "
                "fluid checks, tire service, and component replacement."
            ),
        ),
        embedding=(0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
    ),
)


VEHICLE_EVALUATION_CASES = (
    RetrievalEvaluationCase(
        query="How does an electric vehicle get its power?",
        relevant_chunk_ids=(
            "vehicle-electric-powertrain",
            "vehicle-battery-charging",
            "vehicle-regenerative-braking",
        ),
        relevance_grades={
            "vehicle-electric-powertrain": 3.0,
            "vehicle-battery-charging": 2.0,
            "vehicle-regenerative-braking": 1.0,
        },
    ),
    RetrievalEvaluationCase(
        query="How does a gasoline vehicle generate propulsion?",
        relevant_chunk_ids=(
            "vehicle-gasoline-engine",
            "vehicle-hybrid-powertrain",
        ),
        relevance_grades={
            "vehicle-gasoline-engine": 3.0,
            "vehicle-hybrid-powertrain": 1.0,
        },
    ),
    RetrievalEvaluationCase(
        query="How does a hybrid vehicle combine power sources?",
        relevant_chunk_ids=(
            "vehicle-hybrid-powertrain",
            "vehicle-gasoline-engine",
            "vehicle-electric-powertrain",
        ),
        relevance_grades={
            "vehicle-hybrid-powertrain": 3.0,
            "vehicle-gasoline-engine": 2.0,
            "vehicle-electric-powertrain": 2.0,
        },
    ),
    RetrievalEvaluationCase(
        query="How is an electric vehicle battery charged?",
        relevant_chunk_ids=(
            "vehicle-battery-charging",
            "vehicle-electric-powertrain",
            "vehicle-regenerative-braking",
        ),
        relevance_grades={
            "vehicle-battery-charging": 3.0,
            "vehicle-electric-powertrain": 1.0,
            "vehicle-regenerative-braking": 1.0,
        },
    ),
    RetrievalEvaluationCase(
        query="How does regenerative braking recover energy in an electric vehicle?",
        relevant_chunk_ids=(
            "vehicle-regenerative-braking",
            "vehicle-battery-charging",
            "vehicle-electric-powertrain",
        ),
        relevance_grades={
            "vehicle-regenerative-braking": 3.0,
            "vehicle-battery-charging": 2.0,
            "vehicle-electric-powertrain": 1.0,
        },
    ),
    RetrievalEvaluationCase(
        query="What component stores energy for an electric vehicle?",
        relevant_chunk_ids=(
            "vehicle-battery-charging",
            "vehicle-electric-powertrain",
            "vehicle-regenerative-braking",
        ),
        relevance_grades={
            "vehicle-battery-charging": 3.0,
            "vehicle-electric-powertrain": 2.0,
            "vehicle-regenerative-braking": 1.0,
        },
    ),
    RetrievalEvaluationCase(
        query="What does routine vehicle maintenance involve?",
        relevant_chunk_ids=("vehicle-maintenance",),
        relevance_grades={
            "vehicle-maintenance": 3.0,
        },
    ),
)


def vehicle_benchmark_embedding(text: str) -> EmbeddingResult:
    """
    Deterministic embedding function for the vehicle retrieval benchmark.

    This is intentionally a benchmark fixture rather than a production
    embedding implementation. It creates a controlled semantic space so
    retrieval quality can be measured reproducibly without external model
    dependencies.
    """
    normalized = text.lower()

    vector = [0.0] * 6

    if "electric" in normalized or "ev" in normalized:
        vector[0] = 1.0

    if "gasoline" in normalized or "combustion" in normalized:
        vector[1] = 1.0

    if "hybrid" in normalized:
        vector[0] = 1.0
        vector[1] = 1.0
        vector[2] = 1.0

    if (
        "charg" in normalized
        or "battery" in normalized
        or "store" in normalized
        or "energy" in normalized
    ):
        vector[3] = 1.0

    if "brak" in normalized or "recover" in normalized:
        vector[4] = 1.0

    if (
        "maintenance" in normalized
        or "inspection" in normalized
        or "tire" in normalized
        or "fluid" in normalized
    ):
        vector[5] = 1.0

    return EmbeddingResult(
        vector=tuple(vector),
        identity=VEHICLE_EMBEDDING_IDENTITY,
    )


class VehicleBenchmarkEmbeddingService:
    """
    EmbeddingService-compatible adapter for the vehicle benchmark.
    """

    async def embed(self, text: str) -> tuple[float, ...]:
        return vehicle_benchmark_embedding(text).vector

    async def embed_with_metadata(
        self,
        text: str,
    ) -> EmbeddingResult:
        return vehicle_benchmark_embedding(text)


def vehicle_benchmark_chunks() -> tuple[EmbeddedChunk, ...]:
    """
    Return benchmark chunks with embedding provenance attached.
    """
    return tuple(
        EmbeddedChunk(
            chunk=item.chunk,
            embedding=item.embedding,
            embedding_identity=VEHICLE_EMBEDDING_IDENTITY,
        )
        for item in VEHICLE_BENCHMARK_ITEMS
    )


def vehicle_evaluation_cases() -> tuple[RetrievalEvaluationCase, ...]:
    return VEHICLE_EVALUATION_CASES
