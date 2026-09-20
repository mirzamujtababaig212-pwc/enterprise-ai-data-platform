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
