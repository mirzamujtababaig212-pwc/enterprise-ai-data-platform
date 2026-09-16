from __future__ import annotations

import pytest

from ai_platform.agents.composite_observer import (
    CompositeAgentExecutionObserver,
)
from ai_platform.agents.observability import (
    AgentExecutionEvent,
    AgentExecutionEventType,
)
from ai_platform.agents.observer import AgentExecutionObserver


class FakeAgentExecutionObserver:
    def __init__(self, name: str, events: list[str]) -> None:
        self.name = name
        self.events = events

    async def record(
        self,
        event: AgentExecutionEvent,
    ) -> None:
        self.events.append(f"{self.name}:{event.event_type.value}")


def test_composite_observer_matches_protocol() -> None:
    observer = CompositeAgentExecutionObserver([])

    assert isinstance(
        observer,
        AgentExecutionObserver,
    )


@pytest.mark.asyncio
async def test_composite_observer_forwards_event_to_all_observers_in_order() -> None:
    events: list[str] = []

    first = FakeAgentExecutionObserver("first", events)
    second = FakeAgentExecutionObserver("second", events)

    observer = CompositeAgentExecutionObserver(
        [first, second],
    )

    event = AgentExecutionEvent(
        event_type=AgentExecutionEventType.AGENT_STARTED,
        agent_name="test-agent",
    )

    await observer.record(event)

    assert events == [
        "first:agent.started",
        "second:agent.started",
    ]


@pytest.mark.asyncio
async def test_composite_observer_propagates_observer_failure() -> None:
    class FailingObserver:
        async def record(
            self,
            event: AgentExecutionEvent,
        ) -> None:
            raise RuntimeError("observer failure")

    observer = CompositeAgentExecutionObserver(
        [FailingObserver()],
    )

    event = AgentExecutionEvent(
        event_type=AgentExecutionEventType.AGENT_COMPLETED,
        agent_name="test-agent",
    )

    with pytest.raises(RuntimeError, match="observer failure"):
        await observer.record(event)
