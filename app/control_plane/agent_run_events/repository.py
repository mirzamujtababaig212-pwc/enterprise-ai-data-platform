from __future__ import annotations

from typing import Protocol

from ai_platform.agents.observability import (
    AgentExecutionEvent,
    AgentExecutionEventType,
)

from app.control_plane.agent_run_events.models import AgentRunEventsPage


class AgentRunEventsRepository(Protocol):
    def record(
        self,
        event: AgentExecutionEvent,
        *,
        commit: bool = True,
    ) -> AgentExecutionEvent: ...

    def list_page(
        self,
        run_id: str,
        *,
        event_type: AgentExecutionEventType | None = None,
        step_id: str | None = None,
        attempt: int | None = None,
        provider: str | None = None,
        cursor: str | None = None,
        limit: int = 100,
    ) -> AgentRunEventsPage: ...

    def list(
        self,
        run_id: str,
        *,
        event_type: AgentExecutionEventType | None = None,
        step_id: str | None = None,
        attempt: int | None = None,
        provider: str | None = None,
        cursor: str | None = None,
        limit: int = 100,
    ) -> list[AgentExecutionEvent]: ...
