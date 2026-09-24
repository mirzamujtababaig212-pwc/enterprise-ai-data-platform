from __future__ import annotations

import time

import pytest

from ai_platform.agents.budget import (
    ExecutionBudget,
    ExecutionBudgetState,
)


def test_execution_budget_is_immutable() -> None:
    budget = ExecutionBudget()

    with pytest.raises(AttributeError):
        budget.max_llm_calls = 20  # type: ignore[misc]


def test_execution_budget_state_initializes_started_at() -> None:
    before = time.monotonic()

    state = ExecutionBudgetState()

    after = time.monotonic()

    assert before <= state.started_at <= after


def test_execution_budget_state_accepts_existing_counters() -> None:
    state = ExecutionBudgetState(
        llm_calls=2,
        tool_calls=4,
        tool_rounds=1,
        started_at=100.0,
    )

    assert state.llm_calls == 2
    assert state.tool_calls == 4
    assert state.tool_rounds == 1
    assert state.started_at == 100.0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("llm_calls", -1),
        ("tool_calls", -1),
        ("tool_rounds", -1),
    ],
)
def test_execution_budget_state_rejects_negative_counters(
    field: str,
    value: int,
) -> None:
    with pytest.raises(ValueError):
        ExecutionBudgetState(**{field: value})


def test_execution_budget_state_consumes_llm_calls() -> None:
    from ai_platform.agents.budget import ExecutionBudget, ExecutionBudgetState

    budget = ExecutionBudget(max_llm_calls=2)
    state = ExecutionBudgetState()

    state.consume_llm_call(budget, "test-agent")
    state.consume_llm_call(budget, "test-agent")

    assert state.llm_calls == 2


def test_execution_budget_state_rejects_excess_llm_calls() -> None:
    import pytest

    from ai_platform.agents.budget import ExecutionBudget, ExecutionBudgetState
    from ai_platform.agents.exceptions import AgentLLMCallLimitError

    budget = ExecutionBudget(max_llm_calls=1)
    state = ExecutionBudgetState()

    state.consume_llm_call(budget, "test-agent")

    with pytest.raises(
        AgentLLMCallLimitError,
        match="maximum LLM calls \\(1\\)",
    ):
        state.consume_llm_call(budget, "test-agent")

    assert state.llm_calls == 1


def test_execution_budget_state_consumes_multiple_tool_calls() -> None:
    from ai_platform.agents.budget import ExecutionBudget, ExecutionBudgetState

    budget = ExecutionBudget(max_tool_calls=3)
    state = ExecutionBudgetState()

    state.consume_tool_calls(2, budget, "test-agent")

    assert state.tool_calls == 2


def test_execution_budget_state_rejects_excess_tool_calls() -> None:
    import pytest

    from ai_platform.agents.budget import ExecutionBudget, ExecutionBudgetState
    from ai_platform.agents.exceptions import AgentToolCallLimitError

    budget = ExecutionBudget(max_tool_calls=2)
    state = ExecutionBudgetState()

    state.consume_tool_calls(2, budget, "test-agent")

    with pytest.raises(
        AgentToolCallLimitError,
        match="maximum tool calls \\(2\\)",
    ):
        state.consume_tool_calls(1, budget, "test-agent")

    assert state.tool_calls == 2


def test_execution_budget_state_consumes_tool_rounds() -> None:
    from ai_platform.agents.budget import ExecutionBudget, ExecutionBudgetState

    budget = ExecutionBudget(max_tool_rounds=2)
    state = ExecutionBudgetState()

    state.consume_tool_round(budget, "test-agent")
    state.consume_tool_round(budget, "test-agent")

    assert state.tool_rounds == 2


def test_execution_budget_state_rejects_excess_tool_rounds() -> None:
    import pytest

    from ai_platform.agents.budget import ExecutionBudget, ExecutionBudgetState
    from ai_platform.agents.exceptions import AgentToolLoopLimitError

    budget = ExecutionBudget(max_tool_rounds=1)
    state = ExecutionBudgetState()

    state.consume_tool_round(budget, "test-agent")

    with pytest.raises(
        AgentToolLoopLimitError,
        match="maximum tool-call rounds \\(1\\)",
    ):
        state.consume_tool_round(budget, "test-agent")

    assert state.tool_rounds == 1


def test_execution_budget_state_rejects_negative_tool_call_count() -> None:
    import pytest

    from ai_platform.agents.budget import ExecutionBudget, ExecutionBudgetState

    state = ExecutionBudgetState()

    with pytest.raises(ValueError, match="must not be negative"):
        state.consume_tool_calls(
            -1,
            ExecutionBudget(),
            "test-agent",
        )


def test_execution_budget_state_round_trips_elapsed_time() -> None:
    from ai_platform.agents.budget import ExecutionBudgetState

    state = ExecutionBudgetState(
        llm_calls=3,
        tool_calls=5,
        tool_rounds=2,
    )

    payload = state.to_dict()
    restored = ExecutionBudgetState.from_dict(payload)

    assert restored.llm_calls == 3
    assert restored.tool_calls == 5
    assert restored.tool_rounds == 2
    assert restored.elapsed_seconds >= 0


def test_execution_budget_state_from_dict_defaults_missing_fields() -> None:
    from ai_platform.agents.budget import ExecutionBudgetState

    restored = ExecutionBudgetState.from_dict({})

    assert restored.llm_calls == 0
    assert restored.tool_calls == 0
    assert restored.tool_rounds == 0
    assert restored.elapsed_seconds >= 0


def test_execution_budget_state_rejects_negative_elapsed_time() -> None:
    from ai_platform.agents.budget import ExecutionBudgetState

    with pytest.raises(ValueError, match="elapsed_seconds"):
        ExecutionBudgetState.from_dict({"elapsed_seconds": -1})


def test_execution_budget_state_rejects_invalid_elapsed_time() -> None:
    from ai_platform.agents.budget import ExecutionBudgetState

    with pytest.raises(TypeError, match="elapsed_seconds"):
        ExecutionBudgetState.from_dict({"elapsed_seconds": "invalid"})


def test_execution_budget_state_consumes_tokens() -> None:
    budget = ExecutionBudget(max_tokens_per_run=10_000)
    state = ExecutionBudgetState()

    state.consume_tokens(2_500, budget, "test-agent")
    state.consume_tokens(1_500, budget, "test-agent")

    assert state.total_tokens == 4_000


def test_execution_budget_state_rejects_excess_tokens() -> None:
    from ai_platform.agents.exceptions import AgentTokenLimitError

    budget = ExecutionBudget(max_tokens_per_run=5_000)
    state = ExecutionBudgetState()

    state.consume_tokens(4_000, budget, "test-agent")

    with pytest.raises(
        AgentTokenLimitError,
        match=r"maximum token usage \(5000; actual: 6000\)",
    ):
        state.consume_tokens(2_000, budget, "test-agent")

    # Actual provider-reported usage remains authoritative even when
    # the run exceeds its configured token ceiling.
    assert state.total_tokens == 6_000


def test_execution_budget_state_round_trips_total_tokens() -> None:
    state = ExecutionBudgetState(
        llm_calls=3,
        tool_calls=5,
        tool_rounds=2,
        total_tokens=7_500,
    )

    payload = state.to_dict()
    restored = ExecutionBudgetState.from_dict(payload)

    assert payload["total_tokens"] == 7_500
    assert restored.total_tokens == 7_500


def test_execution_budget_state_from_dict_defaults_missing_total_tokens() -> None:
    restored = ExecutionBudgetState.from_dict(
        {
            "llm_calls": 2,
            "tool_calls": 3,
            "tool_rounds": 1,
        }
    )

    assert restored.total_tokens == 0
