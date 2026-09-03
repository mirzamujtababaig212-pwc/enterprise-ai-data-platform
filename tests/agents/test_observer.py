from __future__ import annotations

import pytest

from ai_platform.agents.observability import (
    AgentExecutionEvent,
    AgentExecutionEventType,
)
from ai_platform.agents.observer import (
    AgentExecutionObserver,
)


class FakeAgentExecutionObserver:
    def __init__(self) -> None:
        self.events: list[AgentExecutionEvent] = []

    async def record(
        self,
        event: AgentExecutionEvent,
    ) -> None:
        self.events.append(event)


@pytest.mark.asyncio
async def test_agent_execution_observer_accepts_execution_event() -> None:
    observer = FakeAgentExecutionObserver()

    event = AgentExecutionEvent(
        event_type=AgentExecutionEventType.AGENT_STARTED,
        agent_name="production-llm-agent",
    )

    await observer.record(event)

    assert observer.events == [event]


def test_fake_observer_matches_agent_execution_observer_protocol() -> None:
    observer = FakeAgentExecutionObserver()

    assert isinstance(
        observer,
        AgentExecutionObserver,
    )
