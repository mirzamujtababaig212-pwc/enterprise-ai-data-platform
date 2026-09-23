from __future__ import annotations

from app.control_plane.mcp_servers.exceptions import (
    MCPServerAlreadyExistsError,
    MCPServerNotFoundError,
)
from app.control_plane.mcp_servers.models import MCPServer
from app.control_plane.mcp_servers.repository import MCPServerRepository


class InMemoryMCPServerRepository(MCPServerRepository):
    def __init__(self) -> None:
        self._servers: dict[str, MCPServer] = {}

    def create(
        self,
        server: MCPServer,
        *,
        commit: bool = True,
    ) -> MCPServer:
        if server.server_id in self._servers:
            raise MCPServerAlreadyExistsError(f"MCP server already exists: {server.server_id}")

        if any(existing.name == server.name for existing in self._servers.values()):
            raise MCPServerAlreadyExistsError(f"MCP server name already exists: {server.name}")

        self._servers[server.server_id] = server
        return server

    def get(self, server_id: str) -> MCPServer | None:
        return self._servers.get(server_id)

    def get_for_tenant(
        self,
        server_id: str,
        tenant_id: str,
    ) -> MCPServer | None:
        server = self._servers.get(server_id)

        if server is None or server.tenant_id != tenant_id:
            return None

        return server

    def get_by_name(self, name: str) -> MCPServer | None:
        return next(
            (server for server in self._servers.values() if server.name == name),
            None,
        )

    def list(
        self,
        *,
        tenant_id: str | None = None,
        desired_state: str | None = None,
        limit: int = 100,
    ) -> list[MCPServer]:
        if limit <= 0:
            raise ValueError("limit must be greater than zero.")

        servers = list(self._servers.values())

        if tenant_id is not None:
            servers = [server for server in servers if server.tenant_id == tenant_id]

        if desired_state is not None:
            servers = [server for server in servers if server.desired_state.value == desired_state]

        servers.sort(
            key=lambda server: (server.created_at, server.server_id),
            reverse=True,
        )

        return servers[:limit]

    def update(
        self,
        server: MCPServer,
        *,
        commit: bool = True,
    ) -> MCPServer:
        if server.server_id not in self._servers:
            raise MCPServerNotFoundError(f"MCP server not found: {server.server_id}")

        existing_name_owner = next(
            (
                existing
                for existing in self._servers.values()
                if existing.name == server.name and existing.server_id != server.server_id
            ),
            None,
        )

        if existing_name_owner is not None:
            raise MCPServerAlreadyExistsError(f"MCP server name already exists: {server.name}")

        self._servers[server.server_id] = server
        return server

    def delete(
        self,
        server_id: str,
        *,
        tenant_id: str,
        commit: bool = True,
    ) -> MCPServer | None:
        server = self.get_for_tenant(server_id, tenant_id)

        if server is None:
            return None

        del self._servers[server_id]
        return server
