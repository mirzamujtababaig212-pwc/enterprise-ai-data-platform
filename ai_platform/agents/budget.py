from __future__ import annotations

from dataclasses import dataclass
from time import monotonic

from ai_platform.agents.exceptions import (
    AgentExecutionDurationLimitError,
    AgentLLMCallLimitError,
    AgentTokenLimitError,
    AgentToolCallLimitError,
    AgentToolLoopLimitError,
)


@dataclass(frozen=True)
class ExecutionBudget:
    """
    Immutable limits for a single agent execution.
    """

    max_llm_calls: int = 10
    max_tool_calls: int = 20
    max_tool_rounds: int = 3
    max_duration_seconds: float = 300.0
    max_tokens_per_run: int | None = None

    def __post_init__(self) -> None:
        if self.max_llm_calls <= 0:
            raise ValueError("Execution max_llm_calls must be greater than zero.")

        if self.max_tool_calls <= 0:
            raise ValueError("Execution max_tool_calls must be greater than zero.")

        if self.max_tool_rounds <= 0:
            raise ValueError("Execution max_tool_rounds must be greater than zero.")

        if self.max_duration_seconds <= 0:
            raise ValueError("Execution max_duration_seconds must be greater than zero.")

        if self.max_tokens_per_run is not None and self.max_tokens_per_run <= 0:
            raise ValueError("Execution max_tokens_per_run must be greater than zero.")


@dataclass
class ExecutionBudgetState:
    """
    Mutable consumption state for one agent execution.
    """

    llm_calls: int = 0
    tool_calls: int = 0
    tool_rounds: int = 0
    started_at: float = 0.0
    total_tokens: int = 0

    def __post_init__(self) -> None:
        if self.llm_calls < 0:
            raise ValueError("Execution llm_calls must not be negative.")

        if self.tool_calls < 0:
            raise ValueError("Execution tool_calls must not be negative.")

        if self.tool_rounds < 0:
            raise ValueError("Execution tool_rounds must not be negative.")

        if self.total_tokens < 0:
            raise ValueError("Execution total_tokens must not be negative.")

        if self.started_at == 0.0:
            self.started_at = monotonic()

    @property
    def elapsed_seconds(self) -> float:
        return monotonic() - self.started_at

    def to_dict(self) -> dict[str, int | float]:
        return {
            "llm_calls": self.llm_calls,
            "tool_calls": self.tool_calls,
            "tool_rounds": self.tool_rounds,
            "total_tokens": self.total_tokens,
            "elapsed_seconds": self.elapsed_seconds,
        }

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, object],
    ) -> "ExecutionBudgetState":
        if not isinstance(payload, dict):
            raise TypeError("Execution budget state payload must be a dictionary.")

        elapsed_seconds = payload.get("elapsed_seconds", 0.0)

        if not isinstance(elapsed_seconds, (int, float)) or isinstance(
            elapsed_seconds,
            bool,
        ):
            raise TypeError("Execution budget elapsed_seconds must be a number.")

        if elapsed_seconds < 0:
            raise ValueError("Execution budget elapsed_seconds must not be negative.")

        return cls(
            llm_calls=int(payload.get("llm_calls", 0)),
            tool_calls=int(payload.get("tool_calls", 0)),
            tool_rounds=int(payload.get("tool_rounds", 0)),
            total_tokens=int(payload.get("total_tokens", 0)),
            started_at=monotonic() - float(elapsed_seconds),
        )

    def consume_tokens(
        self,
        token_count: int,
        budget: ExecutionBudget,
        agent_name: str,
    ) -> None:
        if token_count < 0:
            raise ValueError("Token count must not be negative.")

        self.total_tokens += token_count

        if budget.max_tokens_per_run is not None and self.total_tokens > budget.max_tokens_per_run:
            raise AgentTokenLimitError(
                agent_name,
                budget.max_tokens_per_run,
                self.total_tokens,
            )

    def check_duration(
        self,
        budget: ExecutionBudget,
        agent_name: str,
    ) -> None:
        if self.elapsed_seconds >= budget.max_duration_seconds:
            raise AgentExecutionDurationLimitError(
                agent_name,
                budget.max_duration_seconds,
            )

    def consume_llm_call(
        self,
        budget: ExecutionBudget,
        agent_name: str,
    ) -> None:
        if self.llm_calls >= budget.max_llm_calls:
            raise AgentLLMCallLimitError(
                agent_name,
                budget.max_llm_calls,
            )

        self.llm_calls += 1

    def consume_tool_calls(
        self,
        count: int,
        budget: ExecutionBudget,
        agent_name: str,
    ) -> None:
        if count < 0:
            raise ValueError("Tool call count must not be negative.")

        if self.tool_calls + count > budget.max_tool_calls:
            raise AgentToolCallLimitError(
                agent_name,
                budget.max_tool_calls,
            )

        self.tool_calls += count

    def consume_tool_round(
        self,
        budget: ExecutionBudget,
        agent_name: str,
    ) -> None:
        if self.tool_rounds >= budget.max_tool_rounds:
            raise AgentToolLoopLimitError(
                agent_name,
                budget.max_tool_rounds,
            )

        self.tool_rounds += 1
