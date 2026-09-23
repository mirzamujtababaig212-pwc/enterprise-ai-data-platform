from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from uuid import uuid4

from app.control_plane.mcp_servers.exceptions import MCPServerNotFoundError
from app.control_plane.mcp_servers.models import (
    MCPServer,
    MCPServerDesiredState,
)
from app.control_plane.mcp_servers.repository import MCPServerRepository
from tools.mcp.config import MCPServerConfig
from tools.mcp.manager import MCPServerManager


class MCPServerLifecycleService:
    """
    Tenant-aware lifecycle orchestration for durable MCP server state.

    PostgreSQL/in-memory repositories own durable desired state.
    MCPServerManager owns process-local runtime state.

    These two state stores are intentionally not treated as one atomic
    transaction. Durable desired state is authoritative and runtime state
    is reconciled against it.
    """

    def __init__(
        self,
        *,
        repository: MCPServerRepository,
        manager: MCPServerManager,
    ) -> None:
        self._repository = repository
        self._manager = manager

    def get(
        self,
        *,
        tenant_id: str,
        server_id: str,
    ) -> MCPServer:
        server = self._repository.get_for_tenant(server_id, tenant_id)

        if server is None:
            raise MCPServerNotFoundError(f"MCP server not found for tenant: {server_id}")

        return server

    def list(
        self,
        *,
        tenant_id: str,
        desired_state: MCPServerDesiredState | None = None,
        limit: int = 100,
    ) -> list[MCPServer]:
        if desired_state is not None and not isinstance(
            desired_state,
            MCPServerDesiredState,
        ):
            raise TypeError("desired_state must be an MCPServerDesiredState.")

        return self._repository.list(
            tenant_id=tenant_id,
            desired_state=(desired_state.value if desired_state is not None else None),
            limit=limit,
        )

    async def create(
        self,
        *,
        tenant_id: str,
        config: MCPServerConfig,
        desired_state: MCPServerDesiredState = MCPServerDesiredState.ACTIVE,
        secret_references: dict[str, str] | None = None,
        server_id: str | None = None,
    ) -> MCPServer:
        if not tenant_id.strip():
            raise ValueError("tenant_id must not be empty.")

        if not isinstance(desired_state, MCPServerDesiredState):
            raise TypeError("desired_state must be an MCPServerDesiredState.")

        now = datetime.now(UTC)

        server = MCPServer(
            server_id=server_id or str(uuid4()),
            tenant_id=tenant_id,
            name=config.name,
            desired_state=desired_state,
            config=config,
            secret_references=dict(secret_references or {}),
            created_at=now,
            updated_at=now,
        )

        persisted = self._repository.create(server)

        try:
            await self._reconcile_runtime(persisted)
        except Exception:
            # Durable desired state remains authoritative. Runtime
            # reconciliation can be retried explicitly.
            raise

        return persisted

    async def update(
        self,
        *,
        tenant_id: str,
        server_id: str,
        config: MCPServerConfig | None = None,
        desired_state: MCPServerDesiredState | None = None,
        secret_references: dict[str, str] | None = None,
    ) -> MCPServer:
        existing = self.get(
            tenant_id=tenant_id,
            server_id=server_id,
        )

        if config is not None and config.name != existing.name:
            raise ValueError("MCP server name cannot be changed after creation.")

        updated = replace(
            existing,
            name=config.name if config is not None else existing.name,
            config=config if config is not None else existing.config,
            desired_state=(desired_state if desired_state is not None else existing.desired_state),
            secret_references=(
                dict(secret_references)
                if secret_references is not None
                else dict(existing.secret_references)
            ),
            updated_at=datetime.now(UTC),
        )

        persisted = self._repository.update(updated)

        try:
            await self._reconcile_runtime(persisted)
        except Exception:
            # The durable desired state remains available for later
            # reconciliation if runtime replacement fails.
            raise

        return persisted

    async def set_desired_state(
        self,
        *,
        tenant_id: str,
        server_id: str,
        desired_state: MCPServerDesiredState,
    ) -> MCPServer:
        if not isinstance(desired_state, MCPServerDesiredState):
            raise TypeError("desired_state must be an MCPServerDesiredState.")

        return await self.update(
            tenant_id=tenant_id,
            server_id=server_id,
            desired_state=desired_state,
        )

    async def delete(
        self,
        *,
        tenant_id: str,
        server_id: str,
    ) -> MCPServer:
        existing = self.get(
            tenant_id=tenant_id,
            server_id=server_id,
        )

        if self._is_runtime_registered(existing.name):
            await self._manager.unregister_server(existing.name)

        deleted = self._repository.delete(
            server_id,
            tenant_id=tenant_id,
        )

        if deleted is None:
            raise MCPServerNotFoundError(f"MCP server not found for tenant: {server_id}")

        return deleted

    async def reconcile(
        self,
        *,
        tenant_id: str,
        server_id: str,
    ) -> MCPServer:
        server = self.get(
            tenant_id=tenant_id,
            server_id=server_id,
        )

        await self._reconcile_runtime(server)

        return server

    async def reconcile_tenant(
        self,
        *,
        tenant_id: str,
        limit: int = 100,
    ) -> list[MCPServer]:
        servers = self.list(
            tenant_id=tenant_id,
            limit=limit,
        )

        for server in servers:
            await self._reconcile_runtime(server)

        return servers

    async def _reconcile_runtime(
        self,
        server: MCPServer,
    ) -> None:
        registered = self._is_runtime_registered(server.name)

        if server.desired_state is MCPServerDesiredState.DISABLED:
            if registered:
                await self._manager.unregister_server(server.name)
            return

        if not registered:
            await self._manager.register_server(server.config)
            try:
                await self._manager.connect_and_discover(server.name)
            except Exception:
                # Remove the process-local registration when initial
                # realization fails. Durable desired state remains ACTIVE.
                try:
                    await self._manager.unregister_server(server.name)
                except Exception:
                    pass
                raise
            return

        current_config = self._manager.get_config(server.name)

        if current_config == server.config:
            return

        await self._manager.unregister_server(server.name)

        try:
            await self._manager.register_server(server.config)
            await self._manager.connect_and_discover(server.name)
        except Exception:
            try:
                await self._manager.unregister_server(server.name)
            except Exception:
                pass
            raise

    def _is_runtime_registered(self, name: str) -> bool:
        return name in self._manager.list_servers()
