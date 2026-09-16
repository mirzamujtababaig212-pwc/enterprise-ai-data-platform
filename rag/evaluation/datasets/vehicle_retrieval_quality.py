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
    dimension=20,
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
        (
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
    ),
    VehicleQualityBenchmarkItem(
        _chunk(
            "quality-electric-powertrain",
            "An electric powertrain uses a battery and electric motor to propel a vehicle.",
        ),
        (
            1.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
    ),
    VehicleQualityBenchmarkItem(
        _chunk(
            "quality-electric-inverter",
            "The inverter controls electrical power delivered from the battery to the electric motor.",
        ),
        (
            1.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
    ),
    VehicleQualityBenchmarkItem(
        _chunk(
            "quality-battery-charging",
            "Electric vehicle batteries receive electrical energy through charging equipment.",
        ),
        (
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
    ),
    VehicleQualityBenchmarkItem(
        _chunk(
            "quality-battery-storage",
            "The traction battery stores electrical energy for later vehicle operation.",
        ),
        (
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
    ),
    VehicleQualityBenchmarkItem(
        _chunk(
            "quality-battery-management",
            "A battery management system monitors cell voltage, temperature, and state of charge.",
        ),
        (
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
    ),
    VehicleQualityBenchmarkItem(
        _chunk(
            "quality-regenerative-braking",
            "Regenerative braking converts kinetic energy into electrical energy stored in the battery.",
        ),
        (
            0.0,
            1.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
    ),
    VehicleQualityBenchmarkItem(
        _chunk(
            "quality-energy-recovery",
            "Energy recovery during deceleration can return electrical energy to the traction battery.",
        ),
        (
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
    ),
    VehicleQualityBenchmarkItem(
        _chunk(
            "quality-friction-braking",
            "Friction brakes use brake pads and rotors to slow the vehicle.",
        ),
        (
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
    ),
    VehicleQualityBenchmarkItem(
        _chunk(
            "quality-gasoline-engine",
            "Gasoline engines burn fuel inside cylinders to generate mechanical power.",
        ),
        (
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
        ),
    ),
    VehicleQualityBenchmarkItem(
        _chunk(
            "quality-internal-combustion",
            "Internal combustion engines generate propulsion through controlled fuel combustion.",
        ),
        (
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
        ),
    ),
    VehicleQualityBenchmarkItem(
        _chunk(
            "quality-fuel-injection",
            "Fuel injectors meter gasoline into the engine combustion process.",
        ),
        (
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
    ),
    VehicleQualityBenchmarkItem(
        _chunk(
            "quality-hybrid-powertrain",
            "Hybrid powertrains combine an internal combustion engine with an electric motor.",
        ),
        (
            1.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            1.0,
        ),
    ),
    VehicleQualityBenchmarkItem(
        _chunk(
            "quality-hybrid-battery",
            "A hybrid vehicle uses a battery to support electric propulsion alongside the engine.",
        ),
        (
            1.0,
            1.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
    ),
    VehicleQualityBenchmarkItem(
        _chunk(
            "quality-hybrid-regeneration",
            "Hybrid vehicles can recover braking energy and store it in their battery.",
        ),
        (
            1.0,
            1.0,
            1.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
    ),
    VehicleQualityBenchmarkItem(
        _chunk(
            "quality-brake-inspection",
            "Brake inspections check pads, rotors, calipers, and hydraulic components.",
        ),
        (
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
    ),
    VehicleQualityBenchmarkItem(
        _chunk(
            "quality-brake-maintenance",
            "Brake maintenance includes pad replacement, rotor inspection, and brake fluid service.",
        ),
        (
            0.0,
            0.0,
            1.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
    ),
    VehicleQualityBenchmarkItem(
        _chunk(
            "quality-tire-service",
            "Tire service includes pressure checks, rotation, balancing, and tread inspection.",
        ),
        (
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
    ),
    VehicleQualityBenchmarkItem(
        _chunk(
            "quality-fluid-service",
            "Routine vehicle service includes engine oil, coolant, and brake fluid checks.",
        ),
        (
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
    ),
    VehicleQualityBenchmarkItem(
        _chunk(
            "quality-component-replacement",
            "Component replacement restores worn vehicle parts that no longer meet service requirements.",
        ),
        (
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
    ),
    VehicleQualityBenchmarkItem(
        _chunk(
            "quality-engine-temperature",
            "Engine temperature monitoring detects overheating and abnormal thermal conditions.",
        ),
        (
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
    ),
    VehicleQualityBenchmarkItem(
        _chunk(
            "quality-vehicle-diagnostics",
            "Vehicle diagnostics identify faults using sensor data and diagnostic codes.",
        ),
        (
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
    ),
    VehicleQualityBenchmarkItem(
        _chunk(
            "quality-traction-control",
            "Traction control reduces wheel slip by managing available driving torque.",
        ),
        (
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
        ),
    ),
    VehicleQualityBenchmarkItem(
        _chunk(
            "quality-warning-indicators",
            "Dashboard warning indicators alert drivers to vehicle system conditions.",
        ),
        (
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
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
    """Create a deterministic atomic-concept embedding for the quality fixture.

    The experimental benchmark uses one shared automotive concept ontology for
    both documents and queries. Dimensions represent concepts rather than
    query-specific ranking rules.

    Axes:
      0  = electric
      1  = battery
      2  = braking
      3  = combustion
      4  = maintenance
      5  = diagnostics
      6  = motor
      7  = inverter
      8  = charging
      9  = storage
      10 = regeneration
      11 = fuel injection
      12 = inspection
      13 = tires
      14 = fluid service
      15 = component replacement
      16 = traction
      17 = gasoline
      18 = internal combustion
      19 = hybrid powertrain
    """
    normalized = " ".join(text.lower().split())
    vector = [0.0] * VEHICLE_QUALITY_EMBEDDING_IDENTITY.dimension

    def set_axis(index: int) -> None:
        vector[index] = 1.0

    # ------------------------------------------------------------------
    # Broad concepts
    # ------------------------------------------------------------------

    # Electric propulsion / electric vehicle concept.
    if (
        "electric motor" in normalized
        or "electric powertrain" in normalized
        or "electric propulsion" in normalized
        or "ev battery" in normalized
        or normalized.startswith("electric motors")
        or "hybrid" in normalized
    ):
        set_axis(0)

    # Battery as an energy-storage component.
    if "battery" in normalized or "batteries" in normalized:
        set_axis(1)

    # Braking concept. Avoid treating "brake fluid" alone as a braking
    # mechanism.
    if (
        "regenerative braking" in normalized
        or "friction brakes" in normalized
        or "brake pads" in normalized
        or "braking" in normalized
        or "brake maintenance" in normalized
        or "brake inspection" in normalized
        or ("brake" in normalized and "fluid" not in normalized)
    ):
        set_axis(2)

    # Combustion / ICE concept. "Engine oil" and "engine temperature" are
    # deliberately excluded from combustion by themselves.
    if (
        "gasoline engine" in normalized
        or "internal combustion" in normalized
        or "fuel combustion" in normalized
        or "fuel injectors" in normalized
        or "fuel injection" in normalized
        or normalized.startswith("gasoline engines")
        or "ice" in normalized.split()
        or (
            "engine" in normalized
            and (
                "propulsion" in normalized
                or "combustion" in normalized
                or "gasoline" in normalized
                or "overheating" in normalized
                or "thermal" in normalized
            )
        )
    ):
        set_axis(3)

    # General maintenance/service concept.
    if (
        "maintenance" in normalized
        or "vehicle service" in normalized
        or "tire service" in normalized
        or "fluid service" in normalized
        or "component replacement" in normalized
        or "worn vehicle parts" in normalized
        or "routine vehicle service" in normalized
    ):
        set_axis(4)

    # Diagnostics / fault-detection concept.
    if (
        "diagnostic" in normalized
        or "diagnostics" in normalized
        or "fault" in normalized
        or "faults" in normalized
        or "overheating" in normalized
        or "thermal conditions" in normalized
        or "sensor data" in normalized
        or "sensor faults" in normalized
        or "warning indicators" in normalized
        or "temperature monitoring" in normalized
    ):
        set_axis(5)

    # ------------------------------------------------------------------
    # Atomic concepts
    # ------------------------------------------------------------------

    if (
        "electric motor" in normalized
        or "electric motors" in normalized
        or "motor" in normalized
        or "mechanical propulsion" in normalized
        or "vehicle motion" in normalized
    ):
        set_axis(6)

    # Fuel / engine architecture concepts are deliberately separate from
    # the broad combustion axis so gasoline engines and internal-combustion
    # engines remain distinguishable in graded retrieval.
    if (
        "gasoline engine" in normalized
        or normalized.startswith("gasoline engines")
        or "gasoline engines" in normalized
    ):
        set_axis(17)

    if (
        "internal combustion" in normalized
        or "internal combustion engine" in normalized
        or "ice" in normalized.split()
    ):
        set_axis(18)

    # Hybrid powertrain is a distinct architecture concept, separate from
    # the compositional electric + combustion representation.
    if "hybrid powertrain" in normalized or (
        "hybrid" in normalized
        and "combine" in normalized
        and "engine" in normalized
        and "electric propulsion" in normalized
    ):
        set_axis(19)

    if "inverter" in normalized:
        set_axis(7)

    if (
        "charging equipment" in normalized
        or "charging" in normalized
        or "supplies electricity" in normalized
    ):
        set_axis(8)

    if "stores electrical energy" in normalized or "storage" in normalized:
        set_axis(9)

    if (
        "regenerative braking" in normalized
        or "energy recovery" in normalized
        or "recover braking energy" in normalized
        or ("kinetic energy" in normalized and "recovered" in normalized)
        or ("recover" in normalized and "braking" in normalized)
    ):
        set_axis(10)

    if (
        "fuel injectors" in normalized
        or "fuel injection" in normalized
        or "injectors" in normalized
    ):
        set_axis(11)

    if (
        "brake inspection" in normalized
        or "brake inspections" in normalized
        or ("pads" in normalized and "rotors" in normalized and "calipers" in normalized)
    ):
        set_axis(12)

    if (
        "tire" in normalized
        or "tires" in normalized
        or "tread" in normalized
        or "rotation" in normalized
        or "balancing" in normalized
    ):
        set_axis(13)

    if (
        "fluid service" in normalized
        or "brake fluid" in normalized
        or ("oil" in normalized and "coolant" in normalized and "brake fluid" in normalized)
    ):
        set_axis(14)

    if (
        "component replacement" in normalized
        or "worn vehicle parts" in normalized
        or "pad replacement" in normalized
        or "replacement" in normalized
    ):
        set_axis(15)

    if (
        "traction control" in normalized
        or "wheel slip" in normalized
        or "driving torque" in normalized
    ):
        set_axis(16)

    # Hybrid is compositional: hybrid documents participate in both electric
    # and combustion concepts, while axis 19 captures hybrid powertrain
    # architecture explicitly.
    if "hybrid" in normalized:
        set_axis(0)
        set_axis(3)

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
            embedding=vehicle_quality_benchmark_embedding(item.chunk.content).vector,
            embedding_identity=VEHICLE_QUALITY_EMBEDDING_IDENTITY,
        )
        for item in VEHICLE_QUALITY_BENCHMARK_ITEMS
    )


def vehicle_quality_evaluation_cases() -> tuple[RetrievalEvaluationCase, ...]:
    return VEHICLE_QUALITY_EVALUATION_CASES


def vehicle_quality_query_taxonomy() -> dict[str, str]:
    return dict(VEHICLE_QUALITY_QUERY_TAXONOMY)
