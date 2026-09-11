from __future__ import annotations

from typing import Protocol

from rag.evaluation.run import RetrievalEvaluationRun


class RetrievalEvaluationRunStore(Protocol):
    """Persistence boundary for retrieval evaluation run artifacts."""

    async def save(self, run: RetrievalEvaluationRun) -> None:
        """Persist an evaluation run."""
        ...

    async def get(self, run_id: str) -> RetrievalEvaluationRun | None:
        """Retrieve an evaluation run by ID."""
        ...
