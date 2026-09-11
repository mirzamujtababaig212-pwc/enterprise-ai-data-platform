from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from rag.evaluation.lineage import RetrievalEvaluationLineage
from rag.evaluation.models import RetrievalEvaluationResult
from rag.evaluation.quality_gate import RetrievalQualityGateResult


@dataclass(frozen=True)
class RetrievalEvaluationRun:
    """
    Immutable artifact representing one retrieval evaluation run.

    The run contains evaluation outputs and provenance, but does not contain
    raw query text, retrieved document content, embeddings, or other
    evaluation payloads.
    """

    run_id: str
    created_at: datetime
    lineage: RetrievalEvaluationLineage
    evaluation: RetrievalEvaluationResult
    quality_gate: RetrievalQualityGateResult

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id must not be empty")

        if self.created_at.tzinfo is None:
            raise ValueError("created_at must be timezone-aware")

    @property
    def passed(self) -> bool:
        return self.quality_gate.passed

    def as_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "created_at": self.created_at.isoformat(),
            "lineage": self.lineage.as_dict(),
            "evaluation": self.evaluation.as_dict(),
            "quality_gate": self.quality_gate.as_dict(),
        }
