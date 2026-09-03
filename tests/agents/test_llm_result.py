from __future__ import annotations

from typing import Any

import pytest

from ai_platform.agents.llm_result import (
    AgentLLMResult,
    AgentLLMUsage,
)
from ai_platform.agents.tool_calls import AgentToolCall


def test_llm_usage_defaults_to_zero() -> None:
    usage = AgentLLMUsage()

    assert usage.prompt_tokens == 0
    assert usage.completion_tokens == 0
    assert usage.total_tokens == 0


def test_llm_usage_accepts_token_counts() -> None:
    usage = AgentLLMUsage(
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
    )

    assert usage.prompt_tokens == 10
    assert usage.completion_tokens == 5
    assert usage.total_tokens == 15


@pytest.mark.parametrize(
    "field",
    [
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
    ],
)
def test_llm_usage_rejects_negative_counts(field: str) -> None:
    kwargs: dict[str, int] = {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
    }
    kwargs[field] = -1

    with pytest.raises(
        ValueError,
        match="must be greater than or equal to zero",
    ):
        AgentLLMUsage(**kwargs)


def test_llm_result_contains_normalized_fields() -> None:
    usage = AgentLLMUsage(
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
    )

    result = AgentLLMResult(
        text="Generated answer.",
        provider="openai",
        model="gpt-test",
        usage=usage,
    )

    assert result.text == "Generated answer."
    assert result.provider == "openai"
    assert result.model == "gpt-test"
    assert result.usage == usage
    assert result.raw_response is None


def test_llm_result_can_preserve_raw_response() -> None:
    raw_response: dict[str, Any] = {
        "provider": "openai",
        "model": "gpt-test",
        "reply": "Generated answer.",
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        },
    }

    result = AgentLLMResult(
        text="Generated answer.",
        provider="openai",
        model="gpt-test",
        usage=AgentLLMUsage(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
        ),
        raw_response=raw_response,
    )

    assert result.raw_response == raw_response


@pytest.mark.parametrize(
    "field,value",
    [
        ("text", ""),
        ("text", "   "),
        ("provider", ""),
        ("provider", "   "),
        ("model", ""),
        ("model", "   "),
    ],
)
def test_llm_result_rejects_empty_required_fields(
    field: str,
    value: str,
) -> None:
    kwargs: dict[str, Any] = {
        "text": "Generated answer.",
        "provider": "openai",
        "model": "gpt-test",
        "usage": AgentLLMUsage(),
    }
    kwargs[field] = value

    with pytest.raises(
        ValueError,
        match="must not be empty",
    ):
        AgentLLMResult(**kwargs)


def test_llm_result_is_immutable() -> None:
    result = AgentLLMResult(
        text="Generated answer.",
        provider="openai",
        model="gpt-test",
        usage=AgentLLMUsage(),
    )

    with pytest.raises(AttributeError):
        result.text = "Changed"  # type: ignore[misc]


def test_agent_llm_result_allows_empty_text_with_tool_calls() -> None:
    tool_call = AgentToolCall(
        call_id="call-123",
        name="search",
        arguments={"query": "RAG"},
    )

    result = AgentLLMResult(
        text="",
        provider="openai",
        model="gpt-test",
        usage=AgentLLMUsage(),
        tool_calls=(tool_call,),
    )

    assert result.text == ""
    assert result.tool_calls == (tool_call,)


def test_agent_llm_result_rejects_empty_text_without_tool_calls() -> None:
    with pytest.raises(
        ValueError,
        match="must not be empty",
    ):
        AgentLLMResult(
            text="",
            provider="openai",
            model="gpt-test",
            usage=AgentLLMUsage(),
        )
