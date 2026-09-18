from __future__ import annotations

import pytest

from ai_platform.agents.observability import (
    AgentExecutionEvent,
    AgentExecutionEventType,
)
from ai_platform.agents.observer import AgentExecutionObserver
from app.control_plane.agent_run_events.postgres_observer import (
    PostgreSQLAgentRunEventObserver,
)


def make_event(
    *,
    run_id: str | None = "run-1",
) -> AgentExecutionEvent:
    return AgentExecutionEvent(
        event_type=AgentExecutionEventType.AGENT_STARTED,
        agent_name="vehicle-agent",
        run_id=run_id,
        session_id="session-1",
        metadata={"source": "test"},
    )


class FakeSession:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_observer_matches_protocol() -> None:
    observer = PostgreSQLAgentRunEventObserver(lambda: FakeSession())

    assert isinstance(observer, AgentExecutionObserver)


@pytest.mark.asyncio
async def test_observer_skips_events_without_run_id() -> None:
    created_sessions: list[FakeSession] = []

    def session_factory() -> FakeSession:
        session = FakeSession()
        created_sessions.append(session)
        return session

    observer = PostgreSQLAgentRunEventObserver(session_factory)

    await observer.record(make_event(run_id=None))

    assert created_sessions == []


@pytest.mark.asyncio
async def test_observer_closes_session_after_success(monkeypatch) -> None:
    session = FakeSession()
    recorded: list[AgentExecutionEvent] = []

    class FakeRepository:
        def __init__(self, repository_session) -> None:
            assert repository_session is session

        def record(
            self,
            event: AgentExecutionEvent,
            *,
            commit: bool = True,
        ) -> AgentExecutionEvent:
            recorded.append(event)
            assert commit is True
            return event

    monkeypatch.setattr(
        "app.control_plane.agent_run_events.postgres_observer.PostgreSQLAgentRunEventsRepository",
        FakeRepository,
    )

    observer = PostgreSQLAgentRunEventObserver(lambda: session)
    event = make_event()

    await observer.record(event)

    assert recorded == [event]
    assert session.closed is True


@pytest.mark.asyncio
async def test_observer_closes_session_after_persistence_failure(monkeypatch) -> None:
    session = FakeSession()

    class FailingRepository:
        def __init__(self, repository_session) -> None:
            assert repository_session is session

        def record(
            self,
            event: AgentExecutionEvent,
            *,
            commit: bool = True,
        ) -> AgentExecutionEvent:
            raise RuntimeError("database unavailable")

    monkeypatch.setattr(
        "app.control_plane.agent_run_events.postgres_observer.PostgreSQLAgentRunEventsRepository",
        FailingRepository,
    )

    observer = PostgreSQLAgentRunEventObserver(lambda: session)

    await observer.record(make_event())

    assert session.closed is True


@pytest.mark.asyncio
async def test_observer_creates_fresh_session_for_each_event(monkeypatch) -> None:
    sessions: list[FakeSession] = []

    class FakeRepository:
        def __init__(self, repository_session) -> None:
            self.session = repository_session

        def record(
            self,
            event: AgentExecutionEvent,
            *,
            commit: bool = True,
        ) -> AgentExecutionEvent:
            return event

    monkeypatch.setattr(
        "app.control_plane.agent_run_events.postgres_observer.PostgreSQLAgentRunEventsRepository",
        FakeRepository,
    )

    def session_factory() -> FakeSession:
        session = FakeSession()
        sessions.append(session)
        return session

    observer = PostgreSQLAgentRunEventObserver(session_factory)

    await observer.record(make_event())
    await observer.record(
        AgentExecutionEvent(
            event_type=AgentExecutionEventType.AGENT_COMPLETED,
            agent_name="vehicle-agent",
            run_id="run-1",
        )
    )

    assert len(sessions) == 2
    assert sessions[0] is not sessions[1]
    assert all(session.closed for session in sessions)
