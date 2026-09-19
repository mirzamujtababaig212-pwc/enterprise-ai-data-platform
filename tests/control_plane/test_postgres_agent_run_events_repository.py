from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker

from ai_platform.agents.observability import (
    AgentExecutionEvent,
    AgentExecutionEventType,
)
from app.control_plane.agent_run_events.postgres_repository import (
    PostgreSQLAgentRunEventsRepository,
)
from app.control_plane.persistence.models import Base


def make_event(
    *,
    event_type: AgentExecutionEventType = AgentExecutionEventType.AGENT_STARTED,
    run_id: str | None = "run-1",
    session_id: str | None = "session-1",
    user_id: str | None = "user-1",
    tool_round: int | None = None,
    tool_name: str | None = None,
    call_id: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    metadata: dict | None = None,
) -> AgentExecutionEvent:
    return AgentExecutionEvent(
        event_type=event_type,
        agent_name="vehicle-agent",
        run_id=run_id,
        session_id=session_id,
        user_id=user_id,
        tool_round=tool_round,
        tool_name=tool_name,
        call_id=call_id,
        provider=provider,
        model=model,
        metadata={} if metadata is None else metadata,
    )


def make_repository():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)

    session_factory = sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
    )
    session = session_factory()

    return engine, session, PostgreSQLAgentRunEventsRepository(session)


def test_schema_contains_agent_run_events_table() -> None:
    engine, session, repository = make_repository()

    try:
        assert inspect(repository._session.bind).has_table("agent_run_events")
    finally:
        session.close()
        engine.dispose()


def test_record_and_list_round_trip() -> None:
    engine, session, repository = make_repository()

    try:
        event = make_event(
            event_type=AgentExecutionEventType.LLM_COMPLETED,
            tool_round=1,
            provider="openai",
            model="gpt-4.1-mini",
            metadata={"prompt_tokens": 12, "completion_tokens": 8},
        )

        result = repository.record(event)
        restored = repository.list("run-1")

        assert result == event
        assert len(restored) == 1
        assert restored[0] == event
        assert restored[0].event_type == AgentExecutionEventType.LLM_COMPLETED
        assert restored[0].user_id == "user-1"
        assert restored[0].provider == "openai"
        assert restored[0].model == "gpt-4.1-mini"
        assert restored[0].metadata == {
            "prompt_tokens": 12,
            "completion_tokens": 8,
        }
    finally:
        session.close()
        engine.dispose()


def test_record_preserves_tool_event_fields() -> None:
    engine, session, repository = make_repository()

    try:
        event = make_event(
            event_type=AgentExecutionEventType.TOOL_CALL_COMPLETED,
            tool_round=2,
            tool_name="vehicle_lookup",
            call_id="call-123",
            metadata={"duration_ms": 14},
        )

        repository.record(event)

        restored = repository.list("run-1")

        assert restored == [event]
        assert restored[0].tool_round == 2
        assert restored[0].tool_name == "vehicle_lookup"
        assert restored[0].call_id == "call-123"
    finally:
        session.close()
        engine.dispose()


def test_record_without_run_id_is_skipped() -> None:
    engine, session, repository = make_repository()

    try:
        event = make_event(run_id=None)

        result = repository.record(event)
        restored = repository.list("run-1")

        assert result == event
        assert restored == []
    finally:
        session.close()
        engine.dispose()


def test_record_can_leave_transaction_uncommitted() -> None:
    engine, session, repository = make_repository()

    try:
        event = make_event()

        repository.record(event, commit=False)

        assert repository.list("run-1") == [event]

        session.rollback()

        assert repository.list("run-1") == []
    finally:
        session.close()
        engine.dispose()


def test_list_is_ordered_by_persistence_id() -> None:
    engine, session, repository = make_repository()

    try:
        first = make_event(
            event_type=AgentExecutionEventType.AGENT_STARTED,
            metadata={"sequence": 1},
        )
        second = make_event(
            event_type=AgentExecutionEventType.LLM_REQUESTED,
            tool_round=0,
            metadata={"sequence": 2},
        )
        third = make_event(
            event_type=AgentExecutionEventType.LLM_COMPLETED,
            tool_round=0,
            metadata={"sequence": 3},
        )

        repository.record(first)
        repository.record(second)
        repository.record(third)

        restored = repository.list("run-1")

        assert [event.event_type for event in restored] == [
            AgentExecutionEventType.AGENT_STARTED,
            AgentExecutionEventType.LLM_REQUESTED,
            AgentExecutionEventType.LLM_COMPLETED,
        ]
        assert [event.metadata["sequence"] for event in restored] == [1, 2, 3]
    finally:
        session.close()
        engine.dispose()


def test_list_filters_by_run_id() -> None:
    engine, session, repository = make_repository()

    try:
        repository.record(make_event(run_id="run-1"))
        repository.record(make_event(run_id="run-2"))

        restored = repository.list("run-1")

        assert len(restored) == 1
        assert restored[0].run_id == "run-1"
    finally:
        session.close()
        engine.dispose()


def test_list_respects_limit() -> None:
    engine, session, repository = make_repository()

    try:
        for index in range(5):
            repository.record(
                make_event(
                    metadata={"sequence": index},
                )
            )

        restored = repository.list("run-1", limit=2)

        assert len(restored) == 2
        assert [event.metadata["sequence"] for event in restored] == [0, 1]
    finally:
        session.close()
        engine.dispose()


def test_repository_implements_contract() -> None:
    engine, session, repository = make_repository()

    try:
        from app.control_plane.agent_run_events.repository import (
            AgentRunEventsRepository,
        )

        contract: AgentRunEventsRepository = repository

        assert contract is not None
    finally:
        session.close()
        engine.dispose()
