from __future__ import annotations

import pytest

from ai_platform.agents.tool_calls import (
    AgentToolCall,
    AgentToolResult,
)


def test_agent_tool_call_accepts_valid_call() -> None:
    call = AgentToolCall(
        call_id="call_123",
        name="search",
        arguments={
            "query": "enterprise AI",
        },
    )

    assert call.call_id == "call_123"
    assert call.name == "search"
    assert call.arguments == {
        "query": "enterprise AI",
    }


def test_agent_tool_call_copies_arguments() -> None:
    arguments = {
        "query": "enterprise AI",
    }

    call = AgentToolCall(
        call_id="call_123",
        name="search",
        arguments=arguments,
    )

    arguments["query"] = "changed"

    assert call.arguments == {
        "query": "enterprise AI",
    }


def test_agent_tool_call_allows_empty_arguments() -> None:
    call = AgentToolCall(
        call_id="call_123",
        name="get_status",
        arguments={},
    )

    assert call.arguments == {}


@pytest.mark.parametrize(
    "call_id",
    [
        "",
        "   ",
    ],
)
def test_agent_tool_call_rejects_empty_call_id(
    call_id: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="call_id must not be empty",
    ):
        AgentToolCall(
            call_id=call_id,
            name="search",
            arguments={},
        )


@pytest.mark.parametrize(
    "name",
    [
        "",
        "   ",
    ],
)
def test_agent_tool_call_rejects_empty_name(
    name: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="name must not be empty",
    ):
        AgentToolCall(
            call_id="call_123",
            name=name,
            arguments={},
        )


def test_agent_tool_call_rejects_none_arguments() -> None:
    with pytest.raises(
        ValueError,
        match="arguments must not be None",
    ):
        AgentToolCall(
            call_id="call_123",
            name="search",
            arguments=None,
        )


def test_agent_tool_call_rejects_non_dict_arguments() -> None:
    with pytest.raises(
        TypeError,
        match="arguments must be a dictionary",
    ):
        AgentToolCall(
            call_id="call_123",
            name="search",
            arguments="not-json-object",
        )


def test_agent_tool_result_accepts_success() -> None:
    result = AgentToolResult(
        call_id="call_123",
        tool_name="search",
        output={
            "results": [
                "result-1",
            ],
        },
    )

    assert result.call_id == "call_123"
    assert result.tool_name == "search"
    assert result.output == {
        "results": [
            "result-1",
        ],
    }
    assert result.error is None
    assert result.success is True


def test_agent_tool_result_accepts_failure() -> None:
    result = AgentToolResult(
        call_id="call_123",
        tool_name="search",
        error="Tool execution failed.",
    )

    assert result.output is None
    assert result.error == "Tool execution failed."
    assert result.success is False


@pytest.mark.parametrize(
    "call_id",
    [
        "",
        "   ",
    ],
)
def test_agent_tool_result_rejects_empty_call_id(
    call_id: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="call_id must not be empty",
    ):
        AgentToolResult(
            call_id=call_id,
            tool_name="search",
        )


@pytest.mark.parametrize(
    "tool_name",
    [
        "",
        "   ",
    ],
)
def test_agent_tool_result_rejects_empty_tool_name(
    tool_name: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="tool_name must not be empty",
    ):
        AgentToolResult(
            call_id="call_123",
            tool_name=tool_name,
        )


def test_agent_tool_result_rejects_empty_error() -> None:
    with pytest.raises(
        ValueError,
        match="error must not be empty",
    ):
        AgentToolResult(
            call_id="call_123",
            tool_name="search",
            error="   ",
        )


def test_agent_tool_result_accepts_arbitrary_output() -> None:
    output = {
        "nested": {
            "value": 42,
        },
        "items": [
            1,
            2,
            3,
        ],
    }

    result = AgentToolResult(
        call_id="call_123",
        tool_name="calculator",
        output=output,
    )

    assert result.output == output
    assert result.success is True
