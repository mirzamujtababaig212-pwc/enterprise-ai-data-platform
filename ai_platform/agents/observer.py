from __future__ import annotations

from typing import Protocol, runtime_checkable

from ai_platform.agents.observability import AgentExecutionEvent


@runtime_checkable
class AgentExecutionObserver(Protocol):
    """
    Receives provider-neutral agent execution events.
    """

    async def record(
        self,
        event: AgentExecutionEvent,
    ) -> None: ...
