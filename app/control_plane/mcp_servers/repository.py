from __future__ import annotations

from typing import Protocol

from app.control_plane.mcp_servers.models import MCPServer


class MCPServerRepository(Protocol):
    def create(
        self,
        server: MCPServer,
        *,
        commit: bool = True,
    ) -> MCPServer: ...

    def get(
        self,
        server_id: str,
    ) -> MCPServer | None: ...

    def get_for_tenant(
        self,
        server_id: str,
        tenant_id: str,
    ) -> MCPServer | None: ...

    def get_by_name(
        self,
        name: str,
    ) -> MCPServer | None: ...

    def list(
        self,
        *,
        tenant_id: str | None = None,
        desired_state: str | None = None,
        limit: int = 100,
    ) -> list[MCPServer]: ...

    def update(
        self,
        server: MCPServer,
        *,
        commit: bool = True,
    ) -> MCPServer: ...

    def delete(
        self,
        server_id: str,
        *,
        tenant_id: str,
        commit: bool = True,
    ) -> MCPServer | None: ...
