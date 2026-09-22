from __future__ import annotations

import pytest

from ai_platform.agents.observability import (
    AgentExecutionEvent,
    AgentExecutionEventType,
)


def test_agent_execution_event_accepts_valid_event() -> None:
    event = AgentExecutionEvent(
        event_type=AgentExecutionEventType.LLM_COMPLETED,
        agent_name="production-llm-agent",
        session_id="session-123",
        user_id="user-123",
        tool_round=1,
        provider="openai",
        model="gpt-test",
        metadata={
            "total_tokens": 42,
        },
    )

    assert event.event_type == (AgentExecutionEventType.LLM_COMPLETED)
    assert event.agent_name == "production-llm-agent"
    assert event.session_id == "session-123"
    assert event.user_id == "user-123"
    assert event.tool_round == 1
    assert event.provider == "openai"
    assert event.model == "gpt-test"
    assert event.metadata == {
        "total_tokens": 42,
    }


def test_agent_execution_event_preserves_run_id() -> None:
    event = AgentExecutionEvent(
        event_type=AgentExecutionEventType.AGENT_STARTED,
        agent_name="production-llm-agent",
        run_id="run-123",
    )

    assert event.run_id == "run-123"


@pytest.mark.parametrize("run_id", ["", "   "])
def test_agent_execution_event_rejects_empty_run_id(run_id: str) -> None:
    with pytest.raises(
        ValueError,
        match="Agent execution event run_id must not be empty",
    ):
        AgentExecutionEvent(
            event_type=AgentExecutionEventType.AGENT_STARTED,
            agent_name="production-llm-agent",
            run_id=run_id,
        )


def test_agent_execution_event_rejects_non_string_run_id() -> None:
    with pytest.raises(
        TypeError,
        match="Agent execution event run_id must be a string or None",
    ):
        AgentExecutionEvent(
            event_type=AgentExecutionEventType.AGENT_STARTED,
            agent_name="production-llm-agent",
            run_id=123,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("user_id", ["", "   "])
def test_agent_execution_event_rejects_empty_user_id(user_id: str) -> None:
    with pytest.raises(
        ValueError,
        match="Agent execution event user_id must not be empty",
    ):
        AgentExecutionEvent(
            event_type=AgentExecutionEventType.AGENT_STARTED,
            agent_name="production-llm-agent",
            user_id=user_id,
        )


def test_agent_execution_event_rejects_non_string_user_id() -> None:
    with pytest.raises(
        TypeError,
        match="Agent execution event user_id must be a string or None",
    ):
        AgentExecutionEvent(
            event_type=AgentExecutionEventType.AGENT_STARTED,
            agent_name="production-llm-agent",
            user_id=123,  # type: ignore[arg-type]
        )


def test_agent_execution_event_copies_metadata() -> None:
    metadata = {
        "total_tokens": 42,
    }

    event = AgentExecutionEvent(
        event_type=AgentExecutionEventType.AGENT_COMPLETED,
        agent_name="production-llm-agent",
        metadata=metadata,
    )

    metadata["total_tokens"] = 100

    assert event.metadata == {
        "total_tokens": 42,
    }


def test_agent_execution_event_rejects_empty_agent_name() -> None:
    with pytest.raises(
        ValueError,
        match="agent_name must not be empty",
    ):
        AgentExecutionEvent(
            event_type=AgentExecutionEventType.AGENT_STARTED,
            agent_name=" ",
        )


def test_agent_execution_event_rejects_invalid_event_type() -> None:
    with pytest.raises(
        TypeError,
        match="event_type must be an AgentExecutionEventType",
    ):
        AgentExecutionEvent(
            event_type="agent.started",
            agent_name="production-llm-agent",
        )


def test_agent_execution_event_rejects_negative_tool_round() -> None:
    with pytest.raises(
        ValueError,
        match="tool_round must be >= 0",
    ):
        AgentExecutionEvent(
            event_type=AgentExecutionEventType.TOOL_CALL_REQUESTED,
            agent_name="production-llm-agent",
            tool_round=-1,
        )


def test_agent_execution_event_rejects_empty_tool_name() -> None:
    with pytest.raises(
        ValueError,
        match="tool_name must not be empty",
    ):
        AgentExecutionEvent(
            event_type=AgentExecutionEventType.TOOL_CALL_REQUESTED,
            agent_name="production-llm-agent",
            tool_name=" ",
        )


def test_agent_execution_event_rejects_empty_call_id() -> None:
    with pytest.raises(
        ValueError,
        match="call_id must not be empty",
    ):
        AgentExecutionEvent(
            event_type=AgentExecutionEventType.TOOL_CALL_REQUESTED,
            agent_name="production-llm-agent",
            call_id=" ",
        )


def test_agent_execution_event_rejects_non_dict_metadata() -> None:
    with pytest.raises(
        TypeError,
        match="metadata must be a dictionary",
    ):
        AgentExecutionEvent(
            event_type=AgentExecutionEventType.AGENT_STARTED,
            agent_name="production-llm-agent",
            metadata=[],
        )


@pytest.mark.parametrize(
    ("event_type", "expected_value"),
    [
        (
            AgentExecutionEventType.AGENT_RECOVERY_STARTED,
            "agent.recovery.started",
        ),
        (
            AgentExecutionEventType.AGENT_RECOVERY_COMPLETED,
            "agent.recovery.completed",
        ),
        (
            AgentExecutionEventType.AGENT_RECOVERY_FAILED,
            "agent.recovery.failed",
        ),
    ],
)
def test_agent_recovery_event_types_have_stable_values(
    event_type: AgentExecutionEventType,
    expected_value: str,
) -> None:
    assert event_type.value == expected_value


def test_agent_execution_event_accepts_orchestration_fields() -> None:
    event = AgentExecutionEvent(
        event_type=AgentExecutionEventType.ORCHESTRATION_STEP_STARTED,
        agent_name="enterprise-rag-analyst",
        run_id="run-123",
        step_id="retrieve_evidence",
        step_index=0,
        step_name="Retrieve evidence",
    )

    assert event.step_id == "retrieve_evidence"
    assert event.step_index == 0
    assert event.step_name == "Retrieve evidence"


@pytest.mark.parametrize(
    "field_name,value",
    [
        ("step_id", ""),
        ("step_name", ""),
    ],
)
def test_agent_execution_event_rejects_empty_orchestration_strings(
    field_name,
    value,
) -> None:
    kwargs = {
        "event_type": AgentExecutionEventType.ORCHESTRATION_STEP_STARTED,
        "agent_name": "agent",
        field_name: value,
    }

    with pytest.raises(ValueError):
        AgentExecutionEvent(**kwargs)


def test_agent_execution_event_rejects_negative_orchestration_step_index() -> None:
    with pytest.raises(ValueError, match="step_index"):
        AgentExecutionEvent(
            event_type=AgentExecutionEventType.ORCHESTRATION_STEP_STARTED,
            agent_name="agent",
            step_index=-1,
        )


def test_agent_execution_event_rejects_invalid_orchestration_step_index_type() -> None:
    with pytest.raises(TypeError, match="step_index"):
        AgentExecutionEvent(
            event_type=AgentExecutionEventType.ORCHESTRATION_STEP_STARTED,
            agent_name="agent",
            step_index="0",
        )
