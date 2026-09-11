from __future__ import annotations

from rag.evaluation.run import RetrievalEvaluationRun


class InMemoryRetrievalEvaluationRunStore:
    """Deterministic in-memory store for retrieval evaluation runs."""

    def __init__(self) -> None:
        self._runs: dict[str, RetrievalEvaluationRun] = {}

    async def save(self, run: RetrievalEvaluationRun) -> None:
        self._runs[run.run_id] = run

    async def get(self, run_id: str) -> RetrievalEvaluationRun | None:
        return self._runs.get(run_id)
