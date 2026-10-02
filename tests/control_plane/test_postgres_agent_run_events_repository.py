from datetime import datetime, timedelta

from sqlalchemy import create_engine, inspect, select
from sqlalchemy.orm import sessionmaker

from ai_platform.agents.observability import (
    AgentExecutionEvent,
    AgentExecutionEventType,
)
from app.control_plane.agent_run_events.postgres_repository import (
    PostgreSQLAgentRunEventsRepository,
)
from app.control_plane.persistence.models import (
    AgentRunEventRecord,
    Base,
)


def make_event(
    *,
    event_type: AgentExecutionEventType = AgentExecutionEventType.AGENT_STARTED,
    run_id: str | None = "run-1",
    session_id: str | None = "session-1",
    user_id: str | None = "user-1",
    principal: str | None = "principal-1",
    tool_round: int | None = None,
    tool_name: str | None = None,
    call_id: str | None = None,
    step_id: str | None = None,
    step_index: int | None = None,
    step_name: str | None = None,
    attempt: int | None = None,
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
        principal=principal,
        tool_round=tool_round,
        tool_name=tool_name,
        call_id=call_id,
        step_id=step_id,
        step_index=step_index,
        step_name=step_name,
        attempt=attempt,
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


def set_event_created_at(
    session,
    *,
    sequence: int,
    created_at: datetime,
) -> None:
    record = session.scalar(
        select(AgentRunEventRecord)
        .where(
            AgentRunEventRecord.run_id == "run-1",
            AgentRunEventRecord.event_metadata["sequence"].as_integer() == sequence,
        )
        .order_by(AgentRunEventRecord.id.desc())
    )

    if record is None:
        raise AssertionError(
            f"Event record for sequence {sequence} was not found.",
        )

    record.created_at = created_at
    session.commit()


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
        assert restored[0].principal == "principal-1"
        assert restored[0].provider == "openai"
        assert restored[0].model == "gpt-4.1-mini"
        assert restored[0].metadata == {
            "prompt_tokens": 12,
            "completion_tokens": 8,
        }
    finally:
        session.close()
        engine.dispose()


def test_record_preserves_context_assembly_event_fields() -> None:
    engine, session, repository = make_repository()

    try:
        event = make_event(
            event_type=AgentExecutionEventType.CONTEXT_ASSEMBLY_COMPLETED,
            tool_round=1,
            step_id="retrieve_evidence",
            step_index=0,
            step_name="Retrieve enterprise evidence",
            metadata={
                "total_messages": 7,
                "source_counts": {
                    "system_prompt": 1,
                    "semantic_memory": 3,
                    "chat_history": 1,
                    "user_input": 1,
                    "tool_result": 1,
                },
            },
        )

        repository.record(event)

        restored = repository.list("run-1")

        assert restored == [event]
        assert restored[0].event_type == AgentExecutionEventType.CONTEXT_ASSEMBLY_COMPLETED
        assert restored[0].step_id == "retrieve_evidence"
        assert restored[0].step_index == 0
        assert restored[0].step_name == "Retrieve enterprise evidence"
        assert restored[0].metadata == {
            "total_messages": 7,
            "source_counts": {
                "system_prompt": 1,
                "semantic_memory": 3,
                "chat_history": 1,
                "user_input": 1,
                "tool_result": 1,
            },
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


def test_record_round_trip_preserves_orchestration_attempt_without_metadata_leak() -> None:
    engine, session, repository = make_repository()

    try:
        event = make_event(
            event_type=AgentExecutionEventType.ORCHESTRATION_STEP_FAILED,
            step_id="produce_answer",
            step_index=2,
            step_name="Produce answer",
            attempt=2,
            metadata={
                "retry": {
                    "category": "timeout",
                    "retry_allowed": True,
                }
            },
        )

        repository.record(event)

        restored = repository.list("run-1")

        assert restored == [event]
        assert restored[0].attempt == 2
        assert restored[0].metadata == {
            "retry": {
                "category": "timeout",
                "retry_allowed": True,
            }
        }
        assert "_event_attempt" not in restored[0].metadata
    finally:
        session.close()
        engine.dispose()


def test_record_preserves_orchestration_event_fields() -> None:
    engine, session, repository = make_repository()

    try:
        event = make_event(
            event_type=AgentExecutionEventType.ORCHESTRATION_STEP_COMPLETED,
            step_id="retrieve_evidence",
            step_index=0,
            step_name="Retrieve enterprise evidence",
            metadata={"phase": "evidence_retrieval"},
        )

        repository.record(event)

        restored = repository.list("run-1")

        assert restored == [event]
        assert restored[0].step_id == "retrieve_evidence"
        assert restored[0].step_index == 0
        assert restored[0].step_name == "Retrieve enterprise evidence"
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


def test_list_page_returns_cursor_and_has_more() -> None:
    engine, session, repository = make_repository()

    try:
        base = datetime(2026, 10, 2, 3, 3, 53)

        for index in range(3):
            repository.record(
                make_event(
                    metadata={"sequence": index},
                )
            )
            set_event_created_at(
                session,
                sequence=index,
                created_at=base + timedelta(seconds=index),
            )

        page = repository.list_page("run-1", limit=2)

        assert len(page.events) == 2
        assert [event.metadata["sequence"] for event in page.events] == [0, 1]
        assert page.has_more is True
        assert page.next_cursor is not None
    finally:
        session.close()
        engine.dispose()


def test_list_page_cursor_continues_without_duplicates() -> None:
    engine, session, repository = make_repository()

    try:
        base = datetime(2026, 10, 2, 3, 3, 53)

        for index in range(5):
            repository.record(
                make_event(
                    metadata={"sequence": index},
                )
            )
            set_event_created_at(
                session,
                sequence=index,
                created_at=base + timedelta(seconds=index),
            )

        first_page = repository.list_page("run-1", limit=2)

        second_page = repository.list_page(
            "run-1",
            limit=2,
            cursor=first_page.next_cursor,
        )

        assert [event.metadata["sequence"] for event in first_page.events] == [0, 1]
        assert [event.metadata["sequence"] for event in second_page.events] == [2, 3]
        assert first_page.next_cursor != second_page.next_cursor
        assert second_page.has_more is True
    finally:
        session.close()
        engine.dispose()


def test_list_page_final_page_has_no_cursor() -> None:
    engine, session, repository = make_repository()

    try:
        base = datetime(2026, 10, 2, 3, 3, 53)

        for index in range(3):
            repository.record(
                make_event(
                    metadata={"sequence": index},
                )
            )
            set_event_created_at(
                session,
                sequence=index,
                created_at=base + timedelta(seconds=index),
            )

        first_page = repository.list_page("run-1", limit=2)
        final_page = repository.list_page(
            "run-1",
            limit=2,
            cursor=first_page.next_cursor,
        )

        assert [event.metadata["sequence"] for event in final_page.events] == [2]
        assert final_page.has_more is False
        assert final_page.next_cursor is None
    finally:
        session.close()
        engine.dispose()


def test_list_page_filters_attempt_before_pagination() -> None:
    engine, session, repository = make_repository()

    try:
        repository.record(make_event(attempt=1, metadata={"sequence": 1}))
        repository.record(make_event(attempt=2, metadata={"sequence": 2}))
        repository.record(make_event(attempt=1, metadata={"sequence": 3}))
        repository.record(make_event(attempt=2, metadata={"sequence": 4}))

        page = repository.list_page(
            "run-1",
            attempt=2,
            limit=1,
        )

        assert len(page.events) == 1
        assert page.events[0].attempt == 2
        assert page.events[0].metadata["sequence"] == 2
        assert page.has_more is True
        assert page.next_cursor is not None
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
