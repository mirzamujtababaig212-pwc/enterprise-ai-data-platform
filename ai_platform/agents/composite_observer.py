from __future__ import annotations

from collections.abc import Sequence

from ai_platform.agents.observability import AgentExecutionEvent
from ai_platform.agents.observer import AgentExecutionObserver


class CompositeAgentExecutionObserver(AgentExecutionObserver):
    """
    Forwards provider-neutral agent execution events to multiple observers.

    Observers are invoked sequentially so event ordering is preserved.
    Observer failures are intentionally propagated rather than swallowed.
    """

    def __init__(
        self,
        observers: Sequence[AgentExecutionObserver],
    ) -> None:
        self._observers = tuple(observers)

    async def record(
        self,
        event: AgentExecutionEvent,
    ) -> None:
        for observer in self._observers:
            await observer.record(event)
