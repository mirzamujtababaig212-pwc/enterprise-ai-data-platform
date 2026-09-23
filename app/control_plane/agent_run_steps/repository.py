from __future__ import annotations

from datetime import datetime
from typing import Protocol

from app.control_plane.agent_run_steps.models import (
    AgentRunStep,
    AgentRunStepStatus,
)


class AgentRunStepsRepository(Protocol):
    def close(self) -> None: ...

    def create(
        self,
        step: AgentRunStep,
        *,
        commit: bool = True,
    ) -> AgentRunStep: ...

    def get(
        self,
        run_id: str,
        step_id: str,
    ) -> AgentRunStep | None: ...

    def list(
        self,
        run_id: str,
        *,
        status: AgentRunStepStatus | None = None,
        limit: int = 100,
    ) -> list[AgentRunStep]: ...

    def transition(
        self,
        run_id: str,
        step_id: str,
        *,
        status: AgentRunStepStatus,
        updated_at: datetime,
        started_at: datetime | None = None,
        completed_at: datetime | None = None,
        output=None,
        error: str | None = None,
        failure_category: str | None = None,
        commit: bool = True,
    ) -> AgentRunStep | None: ...

    def bind_execution(
        self,
        run_id: str,
        step_id: str,
        *,
        tool_name: str,
        call_id: str,
        input: object | None = None,
        commit: bool = True,
    ) -> AgentRunStep | None: ...

    def retry(
        self,
        run_id: str,
        step_id: str,
        *,
        updated_at: datetime,
        started_at: datetime | None = None,
        commit: bool = True,
    ) -> AgentRunStep | None: ...
