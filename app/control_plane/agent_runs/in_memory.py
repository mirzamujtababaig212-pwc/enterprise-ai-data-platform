from __future__ import annotations

from datetime import datetime

from threading import Lock

from app.control_plane.agent_runs.exceptions import (
    AgentRunNotFoundError,
    DuplicateAgentRunError,
)
from app.control_plane.agent_runs.models import AgentRun, AgentRunStatus


class InMemoryAgentRunRepository:
    def __init__(self) -> None:
        self._runs: dict[str, AgentRun] = {}
        self._lock = Lock()

    def create(
        self,
        run: AgentRun,
        *,
        commit: bool = True,
    ) -> AgentRun:
        del commit

        with self._lock:
            if run.run_id in self._runs:
                raise DuplicateAgentRunError(f"agent run already exists: {run.run_id}")

            self._runs[run.run_id] = run

        return run

    def get(self, run_id: str) -> AgentRun | None:
        with self._lock:
            return self._runs.get(run_id)

    def update(
        self,
        run: AgentRun,
        *,
        commit: bool = True,
    ) -> AgentRun:
        del commit

        with self._lock:
            if run.run_id not in self._runs:
                raise AgentRunNotFoundError(f"agent run not found: {run.run_id}")

            self._runs[run.run_id] = run

        return run

    def claim_for_recovery(
        self,
        run_id: str,
        *,
        started_at: datetime,
    ) -> AgentRun | None:
        with self._lock:
            run = self._runs.get(run_id)

            if run is None or run.status != AgentRunStatus.FAILED:
                return None

            claimed = run.transition_to(AgentRunStatus.RUNNING).model_copy(
                update={
                    "started_at": started_at,
                    "completed_at": None,
                    "error_type": None,
                    "error_message": None,
                    "output": None,
                }
            )

            self._runs[run_id] = claimed
            return claimed

    def list(
        self,
        *,
        agent_name: str | None = None,
        session_id: str | None = None,
        user_id: str | None = None,
        status: AgentRunStatus | None = None,
        limit: int = 100,
    ) -> list[AgentRun]:
        with self._lock:
            runs = list(self._runs.values())

        if agent_name is not None:
            runs = [run for run in runs if run.agent_name == agent_name]

        if session_id is not None:
            runs = [run for run in runs if run.session_id == session_id]

        if user_id is not None:
            runs = [run for run in runs if run.user_id == user_id]

        if status is not None:
            runs = [run for run in runs if run.status == status]

        return runs[-limit:]

    def clear(self) -> None:
        with self._lock:
            self._runs.clear()
