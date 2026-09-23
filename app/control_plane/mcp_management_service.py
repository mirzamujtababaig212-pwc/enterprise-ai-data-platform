from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ai_platform.agents.policy import (
    TenantPolicyEngine,
    PolicyViolationError,
)


@dataclass(frozen=True)
class MCPServerInfo:
    """Safe control-plane representation of an MCP server."""

    server_id: str
    transport: str
    connected: bool


@dataclass(frozen=True)
class MCPServerHealthInfo:
    """Tenant-scoped MCP server health information."""

    server_id: str
    status: str
    latency_ms: float | None
    last_check: Any
    error: str | None


class MCPManagementService:
    """Tenant-aware control-plane facade over the MCP runtime manager.

    The underlying MCPServerManager remains a global runtime component.
    This service applies tenant policy before exposing server state or
    performing management operations.
    """

    def __init__(
        self,
        *,
        manager: Any,
        tenant_policy_engine: TenantPolicyEngine,
    ) -> None:
        self._manager = manager
        self._tenant_policy_engine = tenant_policy_engine

    async def list_servers(self, tenant_id: str) -> list[MCPServerInfo]:
        """Return MCP servers visible to the tenant.

        An empty allowed_mcp_servers policy preserves the existing policy
        semantics: no MCP-server-specific restriction is applied.
        """

        policy = self._tenant_policy_engine.get_policy(tenant_id)

        servers: list[MCPServerInfo] = []

        for server_id in self._manager.list_servers():
            if policy.allowed_mcp_servers and server_id not in policy.allowed_mcp_servers:
                continue

            config = self._manager.get_config(server_id)

            servers.append(
                MCPServerInfo(
                    server_id=server_id,
                    transport=config.transport,
                    connected=self._manager.is_connected(server_id),
                )
            )

        return servers

    async def health(self, tenant_id: str) -> list[MCPServerHealthInfo]:
        """Return health for MCP servers visible to the tenant.

        Health checks are observational. They deliberately delegate to
        MCPServerManager.check_health(), which does not reconnect or
        rediscover servers.
        """

        policy = self._tenant_policy_engine.get_policy(tenant_id)

        results: list[MCPServerHealthInfo] = []

        for server_id in self._manager.list_servers():
            if policy.allowed_mcp_servers and server_id not in policy.allowed_mcp_servers:
                continue

            health = await self._manager.check_health(server_id)

            status = health.status
            if hasattr(status, "value"):
                status = status.value

            results.append(
                MCPServerHealthInfo(
                    server_id=health.server_name,
                    status=str(status),
                    latency_ms=health.latency_ms,
                    last_check=health.last_check,
                    error=health.error,
                )
            )

        return results

    async def sync(self, tenant_id: str, server_id: str) -> list[Any]:
        """Rediscover tools from an authorized configured MCP server."""

        self._authorize_server(tenant_id, server_id)

        if server_id not in self._manager.list_servers():
            raise KeyError(server_id)

        if not self._manager.is_connected(server_id):
            raise RuntimeError(
                f"MCP server '{server_id}' is not connected.",
            )

        return await self._manager.discover_server(server_id)

    def _authorize_server(self, tenant_id: str, server_id: str) -> None:
        """Apply the tenant's MCP-server allowlist."""

        policy = self._tenant_policy_engine.get_policy(tenant_id)

        if policy.allowed_mcp_servers and server_id not in policy.allowed_mcp_servers:
            raise PolicyViolationError(
                f"MCP server '{server_id}' is not authorized " f"for tenant '{tenant_id}'"
            )
