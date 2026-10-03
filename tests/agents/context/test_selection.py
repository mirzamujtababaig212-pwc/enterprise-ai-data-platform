from __future__ import annotations

import pytest

from ai_platform.agents.context.selection import (
    ContextSelectionPolicy,
    ContextSelector,
    ContextWindowExceededError,
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


def _tool_call_message(
    *,
    call_id: str = "call-1",
    tool_name: str = "search",
) -> AgentMessage:
    return assistant_tool_call_message(
        tool_calls=(
            AgentToolCall(
                call_id=call_id,
                name=tool_name,
                arguments={"query": "test"},
            ),
        ),
    )


def _tool_result_message(
    *,
    call_id: str = "call-1",
    tool_name: str = "search",
) -> AgentMessage:
    return tool_result_message(
        call_id=call_id,
        tool_name=tool_name,
        output={"result": "test"},
    )


def _policy(
    *,
    context_window_tokens: int = 100,
    reserved_output_tokens: int = 20,
) -> ContextSelectionPolicy:
    return ContextSelectionPolicy(
        context_window_tokens=context_window_tokens,
        reserved_output_tokens=reserved_output_tokens,
    )


def test_policy_exposes_input_token_limit() -> None:
    policy = _policy(
        context_window_tokens=100,
        reserved_output_tokens=20,
    )

    assert policy.input_token_limit == 80


def test_policy_rejects_non_positive_context_window() -> None:
    with pytest.raises(ValueError, match="context_window_tokens"):
        ContextSelectionPolicy(
            context_window_tokens=0,
            reserved_output_tokens=0,
        )


def test_policy_rejects_negative_reserved_output_tokens() -> None:
    with pytest.raises(ValueError, match="reserved_output_tokens"):
        ContextSelectionPolicy(
            context_window_tokens=100,
            reserved_output_tokens=-1,
        )


def test_policy_rejects_reserved_output_equal_to_context_window() -> None:
    with pytest.raises(ValueError, match="reserved_output_tokens"):
        ContextSelectionPolicy(
            context_window_tokens=100,
            reserved_output_tokens=100,
        )


def test_under_budget_context_is_preserved() -> None:
    messages = (
        system_message("system"),
        user_message("hello"),
    )

    result = ContextSelector(_policy()).select(messages)

    assert result.messages == messages
    assert result.diagnostics["selection_applied"] is False
    assert result.diagnostics["dropped_message_count"] == 0


def test_oldest_history_is_dropped_first() -> None:
    messages = (
        system_message("system"),
        user_message("old history " * 8),
        assistant_message("old response " * 8),
        user_message("current request"),
    )

    policy = _policy(
        context_window_tokens=40,
        reserved_output_tokens=10,
    )

    result = ContextSelector(policy).select(messages)

    assert result.messages[0] == messages[0]
    assert result.messages[-1] == messages[-1]
    assert len(result.messages) < len(messages)
    assert result.diagnostics["selection_applied"] is True


def test_system_message_is_always_preserved() -> None:
    messages = (
        system_message("system " * 20),
        user_message("old history " * 20),
        assistant_message("old response " * 20),
        user_message("current request"),
    )

    policy = _policy(
        context_window_tokens=50,
        reserved_output_tokens=10,
    )

    result = ContextSelector(policy).select(messages)

    assert result.messages[0].role is AgentMessageRole.SYSTEM
    assert result.messages[0].content == messages[0].content


def test_current_user_message_is_always_preserved() -> None:
    messages = (
        system_message("system"),
        user_message("old history " * 20),
        assistant_message("old response " * 20),
        user_message("current request"),
    )

    policy = _policy(
        context_window_tokens=50,
        reserved_output_tokens=10,
    )

    result = ContextSelector(policy).select(messages)

    assert result.messages[-1].role is AgentMessageRole.USER
    assert result.messages[-1].content == "current request"


def test_tool_call_and_result_are_selected_atomically() -> None:
    tool_call = _tool_call_message()
    tool_result = _tool_result_message()

    messages = (
        system_message("system"),
        user_message("old history " * 20),
        tool_call,
        tool_result,
        user_message("current request"),
    )

    policy = _policy(
        context_window_tokens=70,
        reserved_output_tokens=10,
    )

    result = ContextSelector(policy).select(messages)

    assert (tool_call in result.messages) == (tool_result in result.messages)


def test_multiple_tool_calls_and_results_are_kept_as_one_group() -> None:
    tool_calls = assistant_tool_call_message(
        tool_calls=(
            AgentToolCall(
                call_id="call-1",
                name="search",
                arguments={"query": "one"},
            ),
            AgentToolCall(
                call_id="call-2",
                name="search",
                arguments={"query": "two"},
            ),
        ),
    )

    result_one = tool_result_message(
        call_id="call-1",
        tool_name="search",
        output={"result": "one"},
    )
    result_two = tool_result_message(
        call_id="call-2",
        tool_name="search",
        output={"result": "two"},
    )

    messages = (
        system_message("system"),
        user_message("old history " * 20),
        user_message("current request"),
        tool_calls,
        result_one,
        result_two,
    )

    policy = _policy(
        context_window_tokens=120,
        reserved_output_tokens=10,
    )

    result = ContextSelector(policy).select(messages)

    present = {message.content for message in result.messages}

    tool_group_present = (
        tool_calls.content in present
        and result_one.content in present
        and result_two.content in present
    )

    assert tool_group_present or (
        tool_calls.content not in present
        and result_one.content not in present
        and result_two.content not in present
    )


def test_mandatory_context_overflow_raises() -> None:
    messages = (
        system_message("system " * 50),
        user_message("current request " * 50),
    )

    policy = _policy(
        context_window_tokens=20,
        reserved_output_tokens=5,
    )

    with pytest.raises(ContextWindowExceededError):
        ContextSelector(policy).select(messages)


def test_selection_does_not_mutate_canonical_messages() -> None:
    messages = (
        system_message("system"),
        user_message("old history " * 20),
        assistant_message("old response " * 20),
        user_message("current request"),
    )

    original = tuple(messages)

    policy = _policy(
        context_window_tokens=40,
        reserved_output_tokens=10,
    )

    result = ContextSelector(policy).select(messages)

    assert messages == original
    assert result.messages != messages


def test_selection_is_deterministic() -> None:
    messages = (
        system_message("system"),
        user_message("history one " * 10),
        assistant_message("response one " * 10),
        user_message("history two " * 10),
        assistant_message("response two " * 10),
        user_message("current request"),
    )

    policy = _policy(
        context_window_tokens=80,
        reserved_output_tokens=10,
    )

    selector = ContextSelector(policy)

    first = selector.select(messages)
    second = selector.select(messages)

    assert first.messages == second.messages
    assert first.diagnostics == second.diagnostics


def test_diagnostics_do_not_contain_message_content() -> None:
    secret = "THIS MUST NEVER APPEAR IN DIAGNOSTICS"

    messages = (
        system_message("system"),
        user_message(secret),
    )

    result = ContextSelector(_policy()).select(messages)

    diagnostics_text = repr(result.diagnostics)

    assert secret not in diagnostics_text


def test_diagnostics_report_token_budget_and_selection() -> None:
    messages = (
        system_message("system"),
        user_message("hello"),
    )

    result = ContextSelector(
        _policy(
            context_window_tokens=100,
            reserved_output_tokens=20,
        )
    ).select(messages)

    diagnostics = result.diagnostics

    assert diagnostics["context_window_tokens"] == 100
    assert diagnostics["reserved_output_tokens"] == 20
    assert diagnostics["input_token_limit"] == 80
    assert diagnostics["original_estimated_tokens"] > 0
    assert diagnostics["selected_estimated_tokens"] > 0
    assert diagnostics["dropped_message_count"] == 0
    assert diagnostics["policy_id"] == "default"
    assert diagnostics["policy_version"] == "1.0"


def test_old_tool_group_can_be_dropped_atomically() -> None:
    old_tool_calls = assistant_tool_call_message(
        tool_calls=(
            AgentToolCall(
                call_id="old-call",
                name="search",
                arguments={"query": "old"},
            ),
        ),
    )

    old_tool_result = tool_result_message(
        call_id="old-call",
        tool_name="search",
        output={"result": "old"},
    )

    active_tool_calls = assistant_tool_call_message(
        tool_calls=(
            AgentToolCall(
                call_id="active-call",
                name="search",
                arguments={"query": "active"},
            ),
        ),
    )

    active_tool_result = tool_result_message(
        call_id="active-call",
        tool_name="search",
        output={"result": "active"},
    )

    messages = (
        system_message("system"),
        user_message("old request"),
        old_tool_calls,
        old_tool_result,
        user_message("current request"),
        active_tool_calls,
        active_tool_result,
    )

    result = ContextSelector(
        _policy(
            context_window_tokens=100,
            reserved_output_tokens=10,
        )
    ).select(messages)

    present = {message.content for message in result.messages}

    old_group_present = old_tool_calls.content in present or old_tool_result.content in present

    active_group_present = (
        active_tool_calls.content in present and active_tool_result.content in present
    )

    assert old_group_present is False
    assert active_group_present is True
