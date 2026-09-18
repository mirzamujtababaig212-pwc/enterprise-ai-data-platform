from __future__ import annotations

from typing import Protocol

from ai_platform.agents.observability import AgentExecutionEvent


class AgentRunEventsRepository(Protocol):
    def record(
        self,
        event: AgentExecutionEvent,
        *,
        commit: bool = True,
    ) -> AgentExecutionEvent: ...

    def list(
        self,
        run_id: str,
        *,
        limit: int = 100,
    ) -> list[AgentExecutionEvent]: ...
