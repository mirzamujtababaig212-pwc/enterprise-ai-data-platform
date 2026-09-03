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
