from __future__ import annotations

import pytest

from ai_platform.agents.llm_messages import (
    AgentMessage,
    AgentMessageRole,
    assistant_message,
    assistant_tool_call_message,
    system_message,
    tool_message,
    tool_result_message,
    user_message,
)
from ai_platform.agents.tool_calls import AgentToolCall


def test_agent_message_stores_role_and_content() -> None:
    message = AgentMessage(
        role=AgentMessageRole.USER,
        content="Hello.",
    )

    assert message.role is AgentMessageRole.USER
    assert message.content == "Hello."


def test_agent_message_is_immutable() -> None:
    message = AgentMessage(
        role=AgentMessageRole.USER,
        content="Hello.",
    )

    with pytest.raises(AttributeError):
        message.content = "Changed."


def test_agent_message_rejects_empty_content() -> None:
    with pytest.raises(
        ValueError,
        match="Agent message content must not be empty",
    ):
        AgentMessage(
            role=AgentMessageRole.USER,
            content="   ",
        )


def test_system_message_helper() -> None:
    message = system_message("You are an enterprise agent.")

    assert message == AgentMessage(
        role=AgentMessageRole.SYSTEM,
        content="You are an enterprise agent.",
    )


def test_user_message_helper() -> None:
    message = user_message("Explain RAG.")

    assert message == AgentMessage(
        role=AgentMessageRole.USER,
        content="Explain RAG.",
    )


def test_assistant_message_helper() -> None:
    message = assistant_message("RAG retrieves relevant context.")

    assert message == AgentMessage(
        role=AgentMessageRole.ASSISTANT,
        content="RAG retrieves relevant context.",
    )


def test_tool_message_helper() -> None:
    message = tool_message('{"status": "ok"}')

    assert message == AgentMessage(
        role=AgentMessageRole.TOOL,
        content='{"status": "ok"}',
    )


def test_all_supported_roles_are_available() -> None:
    assert AgentMessageRole.SYSTEM.value == "system"
    assert AgentMessageRole.USER.value == "user"
    assert AgentMessageRole.ASSISTANT.value == "assistant"
    assert AgentMessageRole.TOOL.value == "tool"


def test_tool_result_message_contains_call_identity_and_output() -> None:
    message = tool_result_message(
        call_id="call-123",
        tool_name="search",
        output={
            "results": [
                "document-1",
                "document-2",
            ],
        },
    )

    assert message.role is AgentMessageRole.TOOL

    assert message.content == (
        '{"call_id": "call-123", "output": '
        '{"results": ["document-1", "document-2"]}, '
        '"success": true, "tool_name": "search"}'
    )


def test_tool_result_message_contains_error() -> None:
    message = tool_result_message(
        call_id="call-456",
        tool_name="search",
        error="Search service unavailable.",
    )

    assert message.role is AgentMessageRole.TOOL

    assert message.content == (
        '{"call_id": "call-456", '
        '"error": "Search service unavailable.", '
        '"success": false, "tool_name": "search"}'
    )


def test_tool_result_message_rejects_empty_call_id() -> None:
    with pytest.raises(
        ValueError,
        match="Tool result message call_id must not be empty",
    ):
        tool_result_message(
            call_id=" ",
            tool_name="search",
            output="result",
        )


def test_tool_result_message_rejects_empty_tool_name() -> None:
    with pytest.raises(
        ValueError,
        match="Tool result message tool_name must not be empty",
    ):
        tool_result_message(
            call_id="call-123",
            tool_name=" ",
            output="result",
        )


def test_tool_result_message_rejects_empty_error() -> None:
    with pytest.raises(
        ValueError,
        match="Tool result message error must not be empty",
    ):
        tool_result_message(
            call_id="call-123",
            tool_name="search",
            error=" ",
        )


def test_assistant_tool_call_message_contains_tool_call() -> None:
    message = assistant_tool_call_message(
        tool_calls=(
            AgentToolCall(
                call_id="call-123",
                name="search",
                arguments={
                    "query": "RAG",
                },
            ),
        ),
    )

    assert message.role is AgentMessageRole.ASSISTANT
    assert message.content == (
        '{"tool_calls": [{"arguments": {"query": "RAG"}, '
        '"call_id": "call-123", "name": "search"}]}'
    )


def test_assistant_tool_call_message_preserves_tool_call_order() -> None:
    message = assistant_tool_call_message(
        tool_calls=(
            AgentToolCall(
                call_id="call-1",
                name="search",
                arguments={"query": "RAG"},
            ),
            AgentToolCall(
                call_id="call-2",
                name="status",
                arguments={"service": "pipeline"},
            ),
        ),
    )

    assert message.role is AgentMessageRole.ASSISTANT

    assert message.content == (
        '{"tool_calls": ['
        '{"arguments": {"query": "RAG"}, '
        '"call_id": "call-1", "name": "search"}, '
        '{"arguments": {"service": "pipeline"}, '
        '"call_id": "call-2", "name": "status"}'
        "]}"
    )


def test_assistant_tool_call_message_preserves_content() -> None:
    message = assistant_tool_call_message(
        tool_calls=(
            AgentToolCall(
                call_id="call-123",
                name="search",
                arguments={"query": "RAG"},
            ),
        ),
        content="I will search for the relevant documents.",
    )

    assert message.role is AgentMessageRole.ASSISTANT
    assert message.content == (
        '{"content": "I will search for the relevant documents.", '
        '"tool_calls": [{"arguments": {"query": "RAG"}, '
        '"call_id": "call-123", "name": "search"}]}'
    )


def test_assistant_tool_call_message_rejects_empty_tool_calls() -> None:
    with pytest.raises(
        ValueError,
        match="Assistant tool-call message must contain at least one tool call",
    ):
        assistant_tool_call_message(
            tool_calls=(),
        )


def test_assistant_tool_call_message_rejects_invalid_tool_calls() -> None:
    with pytest.raises(
        TypeError,
        match="Assistant tool-call message must contain AgentToolCall instances",
    ):
        assistant_tool_call_message(
            tool_calls=("invalid",),  # type: ignore[arg-type]
        )
