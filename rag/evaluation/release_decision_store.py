from __future__ import annotations

from typing import Protocol

from rag.evaluation.composite_release import CompositeEvaluationReleaseDecision


class DuplicateEvaluationReleaseDecisionError(ValueError):
    """Raised when a release decision already exists for an evaluation run."""


class RetrievalEvaluationReleaseDecisionStore(Protocol):
    async def save(
        self,
        decision: CompositeEvaluationReleaseDecision,
    ) -> None:
        """Persist a composite release decision."""

    async def get(
        self,
        run_id: str,
    ) -> CompositeEvaluationReleaseDecision | None:
        """Return a persisted decision by evaluation run ID."""
