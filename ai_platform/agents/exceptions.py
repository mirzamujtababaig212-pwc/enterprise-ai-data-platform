from __future__ import annotations


class AgentToolLoopLimitError(RuntimeError):
    """
    Raised when an agent exceeds its maximum allowed tool-call rounds.
    """

    def __init__(
        self,
        agent_name: str,
        max_tool_rounds: int,
    ) -> None:
        if not agent_name.strip():
            raise ValueError("Agent name must not be empty.")

        if max_tool_rounds <= 0:
            raise ValueError("Maximum tool rounds must be greater than zero.")

        self.agent_name = agent_name
        self.max_tool_rounds = max_tool_rounds

        super().__init__(
            f"Agent '{agent_name}' exceeded the maximum " f"tool-call rounds ({max_tool_rounds})."
        )


class AgentExecutionBudgetError(RuntimeError):
    """
    Base exception for execution budget violations.
    """


class AgentLLMCallLimitError(AgentExecutionBudgetError):
    """
    Raised when an agent exceeds its maximum allowed LLM calls.
    """

    def __init__(
        self,
        agent_name: str,
        max_llm_calls: int,
    ) -> None:
        if not agent_name.strip():
            raise ValueError("Agent name must not be empty.")

        if max_llm_calls <= 0:
            raise ValueError("Maximum LLM calls must be greater than zero.")

        self.agent_name = agent_name
        self.max_llm_calls = max_llm_calls

        super().__init__(
            f"Agent '{agent_name}' exceeded the maximum " f"LLM calls ({max_llm_calls})."
        )


class AgentTokenLimitError(AgentExecutionBudgetError):
    """
    Raised when an agent exceeds its maximum allowed token usage.
    """

    def __init__(
        self,
        agent_name: str,
        max_tokens_per_run: int,
        total_tokens: int,
    ) -> None:
        if not agent_name.strip():
            raise ValueError("Agent name must not be empty.")

        if max_tokens_per_run <= 0:
            raise ValueError("Maximum tokens per run must be greater than zero.")

        if total_tokens < 0:
            raise ValueError("Total tokens must not be negative.")

        self.agent_name = agent_name
        self.max_tokens_per_run = max_tokens_per_run
        self.total_tokens = total_tokens

        super().__init__(
            f"Agent '{agent_name}' exceeded the maximum "
            f"token usage ({max_tokens_per_run}; actual: {total_tokens})."
        )


class AgentToolCallLimitError(AgentExecutionBudgetError):
    """
    Raised when an agent exceeds its maximum allowed tool calls.
    """

    def __init__(
        self,
        agent_name: str,
        max_tool_calls: int,
    ) -> None:
        if not agent_name.strip():
            raise ValueError("Agent name must not be empty.")

        if max_tool_calls <= 0:
            raise ValueError("Maximum tool calls must be greater than zero.")

        self.agent_name = agent_name
        self.max_tool_calls = max_tool_calls

        super().__init__(
            f"Agent '{agent_name}' exceeded the maximum " f"tool calls ({max_tool_calls})."
        )


class AgentExecutionDurationLimitError(AgentExecutionBudgetError):
    """
    Raised when an agent exceeds its maximum execution duration.
    """

    def __init__(
        self,
        agent_name: str,
        max_duration_seconds: float,
    ) -> None:
        if not agent_name.strip():
            raise ValueError("Agent name must not be empty.")

        if max_duration_seconds <= 0:
            raise ValueError("Maximum execution duration must be greater than zero.")

        self.agent_name = agent_name
        self.max_duration_seconds = max_duration_seconds

        super().__init__(
            f"Agent '{agent_name}' exceeded the maximum execution "
            f"duration ({max_duration_seconds} seconds)."
        )


class AgentExecutionOwnershipLostError(RuntimeError):
    """Raised when an agent execution loses ownership of its durable run."""
