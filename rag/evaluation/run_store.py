from __future__ import annotations

from typing import Protocol

from rag.evaluation.run import RetrievalEvaluationRun


class DuplicateEvaluationRunError(ValueError):
    """Raised when an evaluation run ID already exists."""


class RetrievalEvaluationRunStore(Protocol):
    """Persistence boundary for retrieval evaluation run artifacts."""

    async def save(self, run: RetrievalEvaluationRun) -> None:
        """Persist an evaluation run."""
        ...

    async def get(self, run_id: str) -> RetrievalEvaluationRun | None:
        """Retrieve an evaluation run by ID."""
        ...

    async def list(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> list[RetrievalEvaluationRun]:
        """List evaluation runs in deterministic order."""
        ...

    async def count(self) -> int:
        """Return the total number of persisted evaluation runs."""
        ...
