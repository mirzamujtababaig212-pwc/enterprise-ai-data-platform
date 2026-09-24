from __future__ import annotations

from threading import Lock

from ai_platform.agents.evaluation.run import AgentEvaluationRun
from app.control_plane.agent_evaluations.repository import (
    DuplicateAgentEvaluationRunError,
)


class InMemoryAgentEvaluationRunsRepository:
    """In-memory repository matching the durable evaluation contract."""

    def __init__(self) -> None:
        self._runs: dict[str, AgentEvaluationRun] = {}
        self._lock = Lock()

    def close(self) -> None:
        return None

    def save(
        self,
        run: AgentEvaluationRun,
        *,
        commit: bool = True,
    ) -> AgentEvaluationRun:
        del commit

        with self._lock:
            if run.evaluation_run_id in self._runs:
                raise DuplicateAgentEvaluationRunError(
                    f"agent evaluation run already exists: " f"{run.evaluation_run_id}"
                )

            self._runs[run.evaluation_run_id] = run

        return run

    def get(
        self,
        evaluation_run_id: str,
    ) -> AgentEvaluationRun | None:
        with self._lock:
            return self._runs.get(evaluation_run_id)

    def list(
        self,
        *,
        evaluated_run_id: str | None = None,
        tenant_id: str | None = None,
        limit: int = 100,
    ) -> list[AgentEvaluationRun]:
        if limit <= 0:
            raise ValueError("limit must be greater than zero.")

        with self._lock:
            runs = [
                run
                for run in self._runs.values()
                if (evaluated_run_id is None or run.lineage.evaluated_run_id == evaluated_run_id)
                and (tenant_id is None or run.lineage.tenant_id == tenant_id)
            ]

            runs.sort(
                key=lambda run: (
                    run.created_at,
                    run.evaluation_run_id,
                ),
                reverse=True,
            )

            return runs[:limit]

    def count(
        self,
        *,
        evaluated_run_id: str | None = None,
        tenant_id: str | None = None,
    ) -> int:
        with self._lock:
            return sum(
                1
                for run in self._runs.values()
                if (evaluated_run_id is None or run.lineage.evaluated_run_id == evaluated_run_id)
                and (tenant_id is None or run.lineage.tenant_id == tenant_id)
            )

    def clear(self) -> None:
        with self._lock:
            self._runs.clear()
