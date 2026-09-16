from __future__ import annotations

from dataclasses import dataclass

from rag.evaluation.models import RetrievalEvaluationCase
from rag.models import (
    DocumentChunk,
    EmbeddedChunk,
    EmbeddingIdentity,
    EmbeddingResult,
)

VEHICLE_QUALITY_QUERY_TAXONOMY = {
    "What converts stored electrical power into vehicle motion?": "semantic",
    "electric inverter battery motor power delivery": "lexical",
    "What equipment supplies electricity to an EV battery?": "mixed",
    "traction battery stores electrical energy": "lexical",
    "How is kinetic energy recovered during braking?": "semantic",
    "regenerative braking battery energy recovery": "lexical",
    "What happens inside an ICE during propulsion?": "semantic",
    "gasoline fuel injectors combustion": "lexical",
    "How does a hybrid combine engine and electric propulsion?": "mixed",
    "pads rotors calipers brake inspection": "lexical",
    "What maintenance replaces worn brake components?": "mixed",
    "tire pressure rotation balancing tread": "lexical",
    "oil coolant brake fluid service": "lexical",
    "Which process restores worn vehicle parts?": "semantic",
    "overheating thermal engine conditions": "lexical",
    "sensor faults diagnostic codes": "lexical",
}

VEHICLE_QUALITY_EMBEDDING_IDENTITY = EmbeddingIdentity(
    requested_provider="test-provider",
    requested_model="vehicle-quality-benchmark",
    resolved_provider="test-provider",
    resolved_model="vehicle-quality-benchmark",
    dimension=8,
)


@dataclass(frozen=True)
class VehicleQualityBenchmarkItem:
    chunk: DocumentChunk
    embedding: tuple[float, ...]


def _chunk(chunk_id: str, content: str) -> DocumentChunk:
    return DocumentChunk(
        id=chunk_id,
        document_id="vehicle-quality-knowledge",
        content=content,
    )


VEHICLE_QUALITY_BENCHMARK_ITEMS = (
    VehicleQualityBenchmarkItem(
        _chunk(
            "quality-electric-motor",
            "Electric motors convert electrical energy into mechanical propulsion.",
        ),
        (1, 0, 0, 0, 0, 0, 0, 0),
    ),
    VehicleQualityBenchmarkItem(
        _chunk(
            "quality-electric-powertrain",
            "An electric powertrain uses a battery and electric motor to propel a vehicle.",
        ),
        (1, 0, 0, 0, 0, 0, 0, 0),
    ),
    VehicleQualityBenchmarkItem(
        _chunk(
            "quality-electric-inverter",
            "The inverter controls electrical power delivered from the battery to the electric motor.",
        ),
        (1, 0, 0, 0, 0, 0, 0, 0),
    ),
    VehicleQualityBenchmarkItem(
        _chunk(
            "quality-battery-charging",
            "Electric vehicle batteries receive electrical energy through charging equipment.",
        ),
        (0, 1, 0, 0, 0, 0, 0, 0),
    ),
    VehicleQualityBenchmarkItem(
        _chunk(
            "quality-battery-storage",
            "The traction battery stores electrical energy for later vehicle operation.",
        ),
        (0, 1, 0, 0, 0, 0, 0, 0),
    ),
    VehicleQualityBenchmarkItem(
        _chunk(
            "quality-battery-management",
            "A battery management system monitors cell voltage, temperature, and state of charge.",
        ),
        (0, 1, 0, 0, 0, 0, 0, 0),
    ),
    VehicleQualityBenchmarkItem(
        _chunk(
            "quality-regenerative-braking",
            "Regenerative braking converts kinetic energy into electrical energy stored in the battery.",
        ),
        (0, 1, 1, 0, 0, 0, 0, 0),
    ),
    VehicleQualityBenchmarkItem(
        _chunk(
            "quality-energy-recovery",
            "Energy recovery during deceleration can return electrical energy to the traction battery.",
        ),
        (0, 1, 1, 0, 0, 0, 0, 0),
    ),
    VehicleQualityBenchmarkItem(
        _chunk(
            "quality-friction-braking",
            "Friction brakes use brake pads and rotors to slow the vehicle.",
        ),
        (0, 0, 1, 0, 0, 0, 0, 0),
    ),
    VehicleQualityBenchmarkItem(
        _chunk(
            "quality-gasoline-engine",
            "Gasoline engines burn fuel inside cylinders to generate mechanical power.",
        ),
        (0, 0, 0, 1, 0, 0, 0, 0),
    ),
    VehicleQualityBenchmarkItem(
        _chunk(
            "quality-internal-combustion",
            "Internal combustion engines generate propulsion through controlled fuel combustion.",
        ),
        (0, 0, 0, 1, 0, 0, 0, 0),
    ),
    VehicleQualityBenchmarkItem(
        _chunk(
            "quality-fuel-injection",
            "Fuel injectors meter gasoline into the engine combustion process.",
        ),
        (0, 0, 0, 1, 0, 0, 0, 0),
    ),
    VehicleQualityBenchmarkItem(
        _chunk(
            "quality-hybrid-powertrain",
            "Hybrid powertrains combine an internal combustion engine with an electric motor.",
        ),
        (1, 0, 0, 1, 0, 0, 0, 0),
    ),
    VehicleQualityBenchmarkItem(
        _chunk(
            "quality-hybrid-battery",
            "A hybrid vehicle uses a battery to support electric propulsion alongside the engine.",
        ),
        (1, 1, 0, 1, 0, 0, 0, 0),
    ),
    VehicleQualityBenchmarkItem(
        _chunk(
            "quality-hybrid-regeneration",
            "Hybrid vehicles can recover braking energy and store it in their battery.",
        ),
        (0, 1, 1, 1, 0, 0, 0, 0),
    ),
    VehicleQualityBenchmarkItem(
        _chunk(
            "quality-brake-inspection",
            "Brake inspections check pads, rotors, calipers, and hydraulic components.",
        ),
        (0, 0, 1, 0, 1, 0, 0, 0),
    ),
    VehicleQualityBenchmarkItem(
        _chunk(
            "quality-brake-maintenance",
            "Brake maintenance includes pad replacement, rotor inspection, and brake fluid service.",
        ),
        (0, 0, 1, 0, 1, 0, 0, 0),
    ),
    VehicleQualityBenchmarkItem(
        _chunk(
            "quality-tire-service",
            "Tire service includes pressure checks, rotation, balancing, and tread inspection.",
        ),
        (0, 0, 0, 0, 1, 0, 0, 0),
    ),
    VehicleQualityBenchmarkItem(
        _chunk(
            "quality-fluid-service",
            "Routine vehicle service includes engine oil, coolant, and brake fluid checks.",
        ),
        (0, 0, 0, 0, 1, 0, 0, 0),
    ),
    VehicleQualityBenchmarkItem(
        _chunk(
            "quality-component-replacement",
            "Component replacement restores worn vehicle parts that no longer meet service requirements.",
        ),
        (0, 0, 0, 0, 1, 0, 0, 0),
    ),
    VehicleQualityBenchmarkItem(
        _chunk(
            "quality-engine-temperature",
            "Engine temperature monitoring detects overheating and abnormal thermal conditions.",
        ),
        (0, 0, 0, 0, 0, 1, 0, 0),
    ),
    VehicleQualityBenchmarkItem(
        _chunk(
            "quality-vehicle-diagnostics",
            "Vehicle diagnostics identify faults using sensor data and diagnostic codes.",
        ),
        (0, 0, 0, 0, 0, 1, 0, 0),
    ),
    VehicleQualityBenchmarkItem(
        _chunk(
            "quality-traction-control",
            "Traction control reduces wheel slip by managing available driving torque.",
        ),
        (1, 0, 0, 0, 0, 0, 1, 0),
    ),
    VehicleQualityBenchmarkItem(
        _chunk(
            "quality-warning-indicators",
            "Dashboard warning indicators alert drivers to vehicle system conditions.",
        ),
        (0, 0, 0, 0, 0, 1, 0, 1),
    ),
)


VEHICLE_QUALITY_EVALUATION_CASES = (
    RetrievalEvaluationCase(
        query="What converts stored electrical power into vehicle motion?",
        relevant_chunk_ids=(
            "quality-electric-motor",
            "quality-electric-powertrain",
        ),
        relevance_grades={
            "quality-electric-motor": 3.0,
            "quality-electric-powertrain": 2.0,
        },
    ),
    RetrievalEvaluationCase(
        query="electric inverter battery motor power delivery",
        relevant_chunk_ids=("quality-electric-inverter",),
        relevance_grades={"quality-electric-inverter": 3.0},
    ),
    RetrievalEvaluationCase(
        query="What equipment supplies electricity to an EV battery?",
        relevant_chunk_ids=("quality-battery-charging",),
        relevance_grades={"quality-battery-charging": 3.0},
    ),
    RetrievalEvaluationCase(
        query="traction battery stores electrical energy",
        relevant_chunk_ids=("quality-battery-storage",),
        relevance_grades={"quality-battery-storage": 3.0},
    ),
    RetrievalEvaluationCase(
        query="How is kinetic energy recovered during braking?",
        relevant_chunk_ids=(
            "quality-regenerative-braking",
            "quality-energy-recovery",
        ),
        relevance_grades={
            "quality-regenerative-braking": 3.0,
            "quality-energy-recovery": 2.0,
        },
    ),
    RetrievalEvaluationCase(
        query="regenerative braking battery energy recovery",
        relevant_chunk_ids=(
            "quality-regenerative-braking",
            "quality-energy-recovery",
            "quality-hybrid-regeneration",
        ),
        relevance_grades={
            "quality-regenerative-braking": 3.0,
            "quality-energy-recovery": 2.0,
            "quality-hybrid-regeneration": 1.0,
        },
    ),
    RetrievalEvaluationCase(
        query="What happens inside an ICE during propulsion?",
        relevant_chunk_ids=(
            "quality-internal-combustion",
            "quality-gasoline-engine",
        ),
        relevance_grades={
            "quality-internal-combustion": 3.0,
            "quality-gasoline-engine": 2.0,
        },
    ),
    RetrievalEvaluationCase(
        query="gasoline fuel injectors combustion",
        relevant_chunk_ids=("quality-fuel-injection",),
        relevance_grades={"quality-fuel-injection": 3.0},
    ),
    RetrievalEvaluationCase(
        query="How does a hybrid combine engine and electric propulsion?",
        relevant_chunk_ids=(
            "quality-hybrid-powertrain",
            "quality-hybrid-battery",
        ),
        relevance_grades={
            "quality-hybrid-powertrain": 3.0,
            "quality-hybrid-battery": 2.0,
        },
    ),
    RetrievalEvaluationCase(
        query="pads rotors calipers brake inspection",
        relevant_chunk_ids=("quality-brake-inspection",),
        relevance_grades={"quality-brake-inspection": 3.0},
    ),
    RetrievalEvaluationCase(
        query="What maintenance replaces worn brake components?",
        relevant_chunk_ids=(
            "quality-brake-maintenance",
            "quality-component-replacement",
        ),
        relevance_grades={
            "quality-brake-maintenance": 3.0,
            "quality-component-replacement": 1.0,
        },
    ),
    RetrievalEvaluationCase(
        query="tire pressure rotation balancing tread",
        relevant_chunk_ids=("quality-tire-service",),
        relevance_grades={"quality-tire-service": 3.0},
    ),
    RetrievalEvaluationCase(
        query="oil coolant brake fluid service",
        relevant_chunk_ids=("quality-fluid-service",),
        relevance_grades={"quality-fluid-service": 3.0},
    ),
    RetrievalEvaluationCase(
        query="Which process restores worn vehicle parts?",
        relevant_chunk_ids=("quality-component-replacement",),
        relevance_grades={"quality-component-replacement": 3.0},
    ),
    RetrievalEvaluationCase(
        query="overheating thermal engine conditions",
        relevant_chunk_ids=("quality-engine-temperature",),
        relevance_grades={"quality-engine-temperature": 3.0},
    ),
    RetrievalEvaluationCase(
        query="sensor faults diagnostic codes",
        relevant_chunk_ids=("quality-vehicle-diagnostics",),
        relevance_grades={"quality-vehicle-diagnostics": 3.0},
    ),
)


def vehicle_quality_benchmark_embedding(text: str) -> EmbeddingResult:
    normalized = text.lower()

    vector = [0.0] * 8

    if any(term in normalized for term in ("electric", "ev", "motor", "propulsion", "hybrid")):
        vector[0] = 1.0

    if any(term in normalized for term in ("battery", "charging", "storage", "energy")):
        vector[1] = 1.0

    if any(term in normalized for term in ("brak", "recover", "deceleration")):
        vector[2] = 1.0

    if any(term in normalized for term in ("gasoline", "combustion", "ice", "fuel", "hybrid")):
        vector[3] = 1.0

    if any(
        term in normalized
        for term in ("maintenance", "service", "inspection", "tire", "fluid", "component")
    ):
        vector[4] = 1.0

    if any(term in normalized for term in ("temperature", "diagnostic", "fault", "warning")):
        vector[5] = 1.0

    if "traction" in normalized or "torque" in normalized:
        vector[6] = 1.0

    if "indicator" in normalized or "dashboard" in normalized:
        vector[7] = 1.0

    return EmbeddingResult(
        vector=tuple(vector),
        identity=VEHICLE_QUALITY_EMBEDDING_IDENTITY,
    )


class VehicleQualityBenchmarkEmbeddingService:
    async def embed(self, text: str) -> tuple[float, ...]:
        return vehicle_quality_benchmark_embedding(text).vector

    async def embed_with_metadata(self, text: str) -> EmbeddingResult:
        return vehicle_quality_benchmark_embedding(text)


def vehicle_quality_benchmark_chunks() -> tuple[EmbeddedChunk, ...]:
    return tuple(
        EmbeddedChunk(
            chunk=item.chunk,
            embedding=item.embedding,
            embedding_identity=VEHICLE_QUALITY_EMBEDDING_IDENTITY,
        )
        for item in VEHICLE_QUALITY_BENCHMARK_ITEMS
    )


def vehicle_quality_evaluation_cases() -> tuple[RetrievalEvaluationCase, ...]:
    return VEHICLE_QUALITY_EVALUATION_CASES


def vehicle_quality_query_taxonomy() -> dict[str, str]:
    return dict(VEHICLE_QUALITY_QUERY_TAXONOMY)
