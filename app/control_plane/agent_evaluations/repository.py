from __future__ import annotations

from typing import Protocol

from ai_platform.agents.evaluation.run import AgentEvaluationRun


class DuplicateAgentEvaluationRunError(ValueError):
    """Raised when an evaluation artifact already exists."""


class AgentEvaluationRunsRepository(Protocol):
    def close(self) -> None: ...

    def save(
        self,
        run: AgentEvaluationRun,
        *,
        commit: bool = True,
    ) -> AgentEvaluationRun: ...

    def get(
        self,
        evaluation_run_id: str,
    ) -> AgentEvaluationRun | None: ...

    def list(
        self,
        *,
        evaluated_run_id: str | None = None,
        tenant_id: str | None = None,
        limit: int = 100,
    ) -> list[AgentEvaluationRun]: ...

    def count(
        self,
        *,
        evaluated_run_id: str | None = None,
        tenant_id: str | None = None,
    ) -> int: ...
