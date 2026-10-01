from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentDelegationPolicy:
    """
    Enterprise policy governing which agents may delegate to which children.

    This policy is intentionally independent from authentication and
    authorization. The caller's identity remains part of the execution
    context; this policy governs delegation semantics.
    """

    max_depth: int = 3
    allowed_child_agents: frozenset[str] | None = None

    def __post_init__(self) -> None:
        if self.max_depth < 1:
            raise ValueError("max_depth must be greater than zero.")

        if self.allowed_child_agents is not None:
            normalized = frozenset(
                name.strip() for name in self.allowed_child_agents if name.strip()
            )
            object.__setattr__(self, "allowed_child_agents", normalized)

    def validate_target(self, child_agent_name: str) -> None:
        if self.allowed_child_agents is None:
            return

        if child_agent_name not in self.allowed_child_agents:
            raise PermissionError(
                "agent delegation target is not permitted: " f"{child_agent_name}"
            )

    def validate_depth(self, depth: int) -> None:
        if depth < 1:
            raise ValueError("delegation depth must be greater than zero.")

        if depth > self.max_depth:
            raise PermissionError(
                "agent delegation depth exceeds configured maximum: " f"{self.max_depth}"
            )
