from __future__ import annotations

import json

import pytest

from ai_platform.agents.checkpoint import (
    AgentCheckpointPosition,
    AgentExecutionCheckpoint,
)
from ai_platform.agents.llm_messages import (
    AgentMessage,
    AgentMessageRole,
    assistant_message,
    assistant_tool_call_message,
    system_message,
    tool_result_message,
    user_message,
)
from ai_platform.agents.tool_calls import AgentToolCall


def test_checkpoint_preserves_execution_identity_and_messages() -> None:
    tool_call = AgentToolCall(
        call_id="call-1",
        name="rag.search",
        arguments={"query": "vehicle warranty"},
    )

    messages = (
        system_message("You are an enterprise AI assistant."),
        user_message("Find the vehicle warranty."),
        assistant_tool_call_message(tool_calls=(tool_call,)),
        tool_result_message(
            call_id="call-1",
            tool_name="rag.search",
            output={"results": [{"document_id": "doc-1"}]},
        ),
    )

    checkpoint = AgentExecutionCheckpoint(
        schema_version=1,
        run_id="run-1",
        agent_name="enterprise-agent",
        session_id="session-1",
        user_id="user-1",
        messages=messages,
        tool_round=1,
        position=AgentCheckpointPosition.AFTER_TOOL_EXECUTION,
        metadata={"provider": "test", "model": "test-model"},
    )

    assert checkpoint.run_id == "run-1"
    assert checkpoint.agent_name == "enterprise-agent"
    assert checkpoint.session_id == "session-1"
    assert checkpoint.user_id == "user-1"
    assert checkpoint.messages == messages
    assert checkpoint.tool_round == 1
    assert checkpoint.metadata == {
        "provider": "test",
        "model": "test-model",
    }


def test_checkpoint_round_trip_preserves_provider_neutral_messages() -> None:
    messages = (
        system_message("System instructions."),
        user_message("Hello."),
        assistant_message("Hello, how can I help?"),
        tool_result_message(
            call_id="call-1",
            tool_name="rag.search",
            output={"answer": "found"},
        ),
    )

    checkpoint = AgentExecutionCheckpoint(
        schema_version=1,
        run_id="run-123",
        agent_name="assistant",
        session_id=None,
        user_id=None,
        messages=messages,
        tool_round=1,
        position=AgentCheckpointPosition.AFTER_TOOL_EXECUTION,
        metadata={"trace": "abc"},
    )

    restored = AgentExecutionCheckpoint.from_json(checkpoint.to_json())

    assert restored == checkpoint
    assert restored.messages == messages


def test_checkpoint_json_is_deterministic_and_json_compatible() -> None:
    checkpoint = AgentExecutionCheckpoint(
        schema_version=1,
        run_id="run-1",
        agent_name="assistant",
        session_id="session-1",
        user_id="user-1",
        messages=(
            AgentMessage(
                role=AgentMessageRole.USER,
                content="What is the status?",
            ),
        ),
        tool_round=0,
        position=AgentCheckpointPosition.AFTER_TOOL_EXECUTION,
        metadata={"b": 2, "a": 1},
    )

    serialized = checkpoint.to_json()

    assert json.loads(serialized) == checkpoint.to_dict()
    assert serialized == checkpoint.to_json()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("run_id", ""),
        ("agent_name", ""),
    ],
)
def test_checkpoint_rejects_empty_required_identity(
    field: str,
    value: str,
) -> None:
    kwargs = {
        "schema_version": 1,
        "run_id": "run-1",
        "agent_name": "assistant",
        "session_id": None,
        "user_id": None,
        "messages": (),
        "tool_round": 0,
        "position": AgentCheckpointPosition.AFTER_TOOL_EXECUTION,
        "metadata": {},
    }
    kwargs[field] = value

    with pytest.raises(ValueError):
        AgentExecutionCheckpoint(**kwargs)


def test_checkpoint_uses_current_schema_version() -> None:
    checkpoint = AgentExecutionCheckpoint(
        schema_version=1,
        run_id="run-1",
        agent_name="assistant",
        session_id=None,
        user_id=None,
        messages=(),
        tool_round=0,
        position=AgentCheckpointPosition.AFTER_TOOL_EXECUTION,
        metadata={},
    )

    assert checkpoint.schema_version == 1
    assert checkpoint.to_dict()["schema_version"] == 1


def test_checkpoint_round_trip_preserves_position() -> None:
    checkpoint = AgentExecutionCheckpoint(
        schema_version=1,
        run_id="run-1",
        agent_name="assistant",
        session_id=None,
        user_id=None,
        messages=(),
        tool_round=1,
        position=AgentCheckpointPosition.AFTER_TOOL_EXECUTION,
        metadata={},
    )

    restored = AgentExecutionCheckpoint.from_json(checkpoint.to_json())

    assert restored.position is AgentCheckpointPosition.AFTER_TOOL_EXECUTION


@pytest.mark.parametrize(
    "schema_version",
    [0, 2, 999],
)
def test_checkpoint_rejects_unsupported_schema_version(
    schema_version: int,
) -> None:
    with pytest.raises(ValueError, match="schema_version"):
        AgentExecutionCheckpoint(
            schema_version=schema_version,
            run_id="run-1",
            agent_name="assistant",
            session_id=None,
            user_id=None,
            messages=(),
            tool_round=0,
            position=AgentCheckpointPosition.AFTER_TOOL_EXECUTION,
            metadata={},
        )


def test_checkpoint_rejects_unsupported_position() -> None:
    with pytest.raises(ValueError, match="after tool execution"):
        AgentExecutionCheckpoint(
            schema_version=1,
            run_id="run-1",
            agent_name="assistant",
            session_id=None,
            user_id=None,
            messages=(),
            tool_round=0,
            position=AgentCheckpointPosition.BEFORE_LLM_REQUEST,
            metadata={},
        )


def test_checkpoint_rejects_negative_tool_round() -> None:
    with pytest.raises(ValueError, match="tool_round"):
        AgentExecutionCheckpoint(
            schema_version=1,
            run_id="run-1",
            agent_name="assistant",
            session_id=None,
            user_id=None,
            messages=(),
            tool_round=-1,
            position=AgentCheckpointPosition.AFTER_TOOL_EXECUTION,
            metadata={},
        )


def test_checkpoint_rejects_invalid_message() -> None:
    with pytest.raises(TypeError, match="AgentMessage"):
        AgentExecutionCheckpoint(
            schema_version=1,
            run_id="run-1",
            agent_name="assistant",
            session_id=None,
            user_id=None,
            messages=("not-a-message",),
            tool_round=0,
            position=AgentCheckpointPosition.AFTER_TOOL_EXECUTION,
            metadata={},
        )
