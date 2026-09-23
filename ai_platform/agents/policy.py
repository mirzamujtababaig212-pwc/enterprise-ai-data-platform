from __future__ import annotations

from dataclasses import dataclass, field
from typing import FrozenSet


class PolicyViolationError(RuntimeError):
    """Raised when an agent tool execution violates tenant policy."""


@dataclass(frozen=True)
class TenantPolicy:
    """Immutable execution policy for a single tenant."""

    tenant_id: str
    allowed_mcp_servers: FrozenSet[str] = field(default_factory=frozenset)
    allowed_tools: FrozenSet[str] = field(default_factory=frozenset)
    blocked_tools: FrozenSet[str] = field(default_factory=frozenset)
    max_tokens_per_run: int | None = None
    allow_cross_tenant_data: bool = False

    def __post_init__(self) -> None:
        if not self.tenant_id:
            raise ValueError("tenant_id must not be empty")

        if self.max_tokens_per_run is not None and self.max_tokens_per_run < 0:
            raise ValueError("max_tokens_per_run must be non-negative")


class TenantPolicyEngine:
    """Evaluates tenant-scoped policy before tool execution."""

    def __init__(self) -> None:
        self._policies: dict[str, TenantPolicy] = {}

    def register_policy(self, policy: TenantPolicy) -> None:
        self._policies[policy.tenant_id] = policy

    def get_policy(self, tenant_id: str) -> TenantPolicy:
        try:
            return self._policies[tenant_id]
        except KeyError as exc:
            raise PolicyViolationError(f"No policy registered for tenant '{tenant_id}'") from exc

    def validate_tool_execution(
        self,
        tenant_id: str,
        tool_name: str,
        server_id: str | None = None,
    ) -> None:
        policy = self.get_policy(tenant_id)

        if tool_name in policy.blocked_tools:
            raise PolicyViolationError(
                f"Tool '{tool_name}' is explicitly blocked " f"for tenant '{tenant_id}'"
            )

        if policy.allowed_tools and tool_name not in policy.allowed_tools:
            raise PolicyViolationError(
                f"Tool '{tool_name}' is not allowed " f"for tenant '{tenant_id}'"
            )

        if server_id and policy.allowed_mcp_servers and server_id not in policy.allowed_mcp_servers:
            raise PolicyViolationError(
                f"MCP server '{server_id}' is not authorized " f"for tenant '{tenant_id}'"
            )
