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
    assert event.tool_round == 1
    assert event.provider == "openai"
    assert event.model == "gpt-test"
    assert event.metadata == {
        "total_tokens": 42,
    }


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
