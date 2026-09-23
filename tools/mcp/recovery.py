from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MCPRecoveryPolicy:
    """
    Bounded recovery policy for an MCP server.

    The policy is deliberately decision-only. It does not perform
    reconnection or mutate MCP runtime state.
    """

    max_attempts: int = 3
    initial_backoff: float = 1.0
    max_backoff: float = 30.0
    cooldown: float = 60.0

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("MCP recovery max_attempts must be at least one.")

        if self.initial_backoff < 0:
            raise ValueError("MCP recovery initial_backoff must be greater than or equal to zero.")

        if self.max_backoff < 0:
            raise ValueError("MCP recovery max_backoff must be greater than or equal to zero.")

        if self.max_backoff < self.initial_backoff:
            raise ValueError(
                "MCP recovery max_backoff must be greater than or equal to " "initial_backoff."
            )

        if self.cooldown < 0:
            raise ValueError("MCP recovery cooldown must be greater than or equal to zero.")

    def backoff_for_attempt(self, attempt: int) -> float:
        """
        Return the bounded exponential backoff for a 1-based attempt.
        """
        if attempt < 1:
            raise ValueError("MCP recovery attempt must be at least one.")

        return min(
            self.initial_backoff * (2 ** (attempt - 1)),
            self.max_backoff,
        )

    def allows_attempt(self, attempt: int) -> bool:
        """
        Return whether the supplied 1-based attempt is within the budget.
        """
        if attempt < 1:
            raise ValueError("MCP recovery attempt must be at least one.")

        return attempt <= self.max_attempts
