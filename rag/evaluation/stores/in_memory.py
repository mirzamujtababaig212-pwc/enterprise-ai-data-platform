from __future__ import annotations

from rag.evaluation.composite_release import CompositeEvaluationReleaseDecision
from rag.evaluation.release_decision_store import (
    DuplicateEvaluationReleaseDecisionError,
)
from rag.evaluation.run import RetrievalEvaluationRun
from rag.evaluation.run_store import DuplicateEvaluationRunError


class InMemoryRetrievalEvaluationRunStore:
    """Deterministic in-memory store for retrieval evaluation runs."""

    def __init__(self) -> None:
        self._runs: dict[str, RetrievalEvaluationRun] = {}

    async def save(self, run: RetrievalEvaluationRun) -> None:
        if run.run_id in self._runs:
            raise DuplicateEvaluationRunError(f"evaluation run already exists: {run.run_id}")

        self._runs[run.run_id] = run

    async def get(self, run_id: str) -> RetrievalEvaluationRun | None:
        return self._runs.get(run_id)

    async def list(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> list[RetrievalEvaluationRun]:
        runs = sorted(
            self._runs.values(),
            key=lambda run: (run.created_at, run.run_id),
            reverse=True,
        )
        return runs[slice(offset, offset + limit)]

    async def count(self) -> int:
        return len(self._runs)


class InMemoryRetrievalEvaluationReleaseDecisionStore:
    """Deterministic in-memory store for composite release decisions."""

    def __init__(self) -> None:
        self._decisions: dict[str, CompositeEvaluationReleaseDecision] = {}

    async def save(
        self,
        decision: CompositeEvaluationReleaseDecision,
    ) -> None:
        if decision.run_id in self._decisions:
            raise DuplicateEvaluationReleaseDecisionError(
                "evaluation release decision already exists: " f"{decision.run_id}"
            )

        self._decisions[decision.run_id] = decision

    async def get(
        self,
        run_id: str,
    ) -> CompositeEvaluationReleaseDecision | None:
        return self._decisions.get(run_id)
