from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest

from app.control_plane.mcp_servers.in_memory import InMemoryMCPServerRepository
from app.control_plane.mcp_servers.lifecycle_service import (
    MCPServerLifecycleService,
)
from app.control_plane.mcp_servers.models import MCPServerDesiredState
from tools.mcp.config import MCPServerConfig


@dataclass
class FakeRuntimeManager:
    registered: dict[str, MCPServerConfig]

    def __init__(self) -> None:
        self.registered = {}
        self.register_server = AsyncMock(side_effect=self._register)
        self.connect_and_discover = AsyncMock()
        self.unregister_server = AsyncMock(side_effect=self._unregister)
        self.get_config = lambda name: self.registered[name]
        self.list_servers = lambda: list(self.registered)

    async def _register(self, config: MCPServerConfig) -> None:
        if config.name in self.registered:
            raise ValueError(f"server already registered: {config.name}")
        self.registered[config.name] = config

    async def _unregister(self, name: str) -> None:
        self.registered.pop(name, None)


def make_config(
    name: str = "document-server",
    *,
    command: str = "document-mcp",
) -> MCPServerConfig:
    return MCPServerConfig(
        name=name,
        transport="stdio",
        command=command,
    )


def make_service() -> tuple[
    MCPServerLifecycleService,
    InMemoryMCPServerRepository,
    FakeRuntimeManager,
]:
    repository = InMemoryMCPServerRepository()
    manager = FakeRuntimeManager()

    service = MCPServerLifecycleService(
        repository=repository,
        manager=manager,
    )

    return service, repository, manager


@pytest.mark.asyncio
async def test_create_active_server_persists_and_realizes_runtime():
    service, repository, manager = make_service()

    server = await service.create(
        tenant_id="tenant-a",
        config=make_config(),
        server_id="server-a",
    )

    assert server.server_id == "server-a"
    assert server.tenant_id == "tenant-a"
    assert server.desired_state is MCPServerDesiredState.ACTIVE

    assert repository.get("server-a") == server
    assert manager.list_servers() == ["document-server"]
    manager.register_server.assert_awaited_once()
    manager.connect_and_discover.assert_awaited_once_with("document-server")


@pytest.mark.asyncio
async def test_create_disabled_server_persists_without_registering_runtime():
    service, repository, manager = make_service()

    server = await service.create(
        tenant_id="tenant-a",
        config=make_config(),
        desired_state=MCPServerDesiredState.DISABLED,
        server_id="server-a",
    )

    assert server.desired_state is MCPServerDesiredState.DISABLED
    assert repository.get("server-a") == server
    assert manager.list_servers() == []
    manager.register_server.assert_not_awaited()
    manager.connect_and_discover.assert_not_awaited()


@pytest.mark.asyncio
async def test_get_is_tenant_scoped():
    service, _, _ = make_service()

    await service.create(
        tenant_id="tenant-a",
        config=make_config(),
        desired_state=MCPServerDesiredState.DISABLED,
        server_id="server-a",
    )

    assert (
        service.get(
            tenant_id="tenant-a",
            server_id="server-a",
        ).tenant_id
        == "tenant-a"
    )

    with pytest.raises(Exception, match="not found"):
        service.get(
            tenant_id="tenant-b",
            server_id="server-a",
        )


@pytest.mark.asyncio
async def test_list_is_tenant_scoped():
    service, _, _ = make_service()

    await service.create(
        tenant_id="tenant-a",
        config=make_config("server-a"),
        desired_state=MCPServerDesiredState.DISABLED,
        server_id="id-a",
    )
    await service.create(
        tenant_id="tenant-b",
        config=make_config("server-b"),
        desired_state=MCPServerDesiredState.DISABLED,
        server_id="id-b",
    )

    servers = service.list(tenant_id="tenant-a")

    assert [server.server_id for server in servers] == ["id-a"]


@pytest.mark.asyncio
async def test_set_desired_state_disabled_unregisters_runtime():
    service, repository, manager = make_service()

    await service.create(
        tenant_id="tenant-a",
        config=make_config(),
        server_id="server-a",
    )

    updated = await service.set_desired_state(
        tenant_id="tenant-a",
        server_id="server-a",
        desired_state=MCPServerDesiredState.DISABLED,
    )

    assert updated.desired_state is MCPServerDesiredState.DISABLED
    assert repository.get("server-a") == updated
    assert manager.list_servers() == []
    manager.unregister_server.assert_awaited_once_with("document-server")


@pytest.mark.asyncio
async def test_set_desired_state_active_registers_disabled_server():
    service, repository, manager = make_service()

    await service.create(
        tenant_id="tenant-a",
        config=make_config(),
        desired_state=MCPServerDesiredState.DISABLED,
        server_id="server-a",
    )

    updated = await service.set_desired_state(
        tenant_id="tenant-a",
        server_id="server-a",
        desired_state=MCPServerDesiredState.ACTIVE,
    )

    assert updated.desired_state is MCPServerDesiredState.ACTIVE
    assert repository.get("server-a") == updated
    assert manager.list_servers() == ["document-server"]
    manager.register_server.assert_awaited_once()
    manager.connect_and_discover.assert_awaited_once_with("document-server")


@pytest.mark.asyncio
async def test_update_replaces_runtime_when_configuration_changes():
    service, repository, manager = make_service()

    await service.create(
        tenant_id="tenant-a",
        config=make_config(command="document-mcp-v1"),
        server_id="server-a",
    )

    updated = await service.update(
        tenant_id="tenant-a",
        server_id="server-a",
        config=make_config(command="document-mcp-v2"),
    )

    assert updated.config.command == "document-mcp-v2"
    assert repository.get("server-a") == updated
    assert manager.registered["document-server"].command == "document-mcp-v2"

    assert manager.unregister_server.await_count == 1
    assert manager.register_server.await_count == 2
    assert manager.connect_and_discover.await_count == 2


@pytest.mark.asyncio
async def test_update_rejects_runtime_server_name_change():
    service, repository, manager = make_service()

    await service.create(
        tenant_id="tenant-a",
        config=make_config("document-server"),
        server_id="server-a",
    )

    with pytest.raises(ValueError, match="server name cannot be changed"):
        await service.update(
            tenant_id="tenant-a",
            server_id="server-a",
            config=make_config("renamed-server"),
        )

    persisted = repository.get("server-a")

    assert persisted is not None
    assert persisted.name == "document-server"
    assert persisted.config.name == "document-server"
    assert manager.list_servers() == ["document-server"]


@pytest.mark.asyncio
async def test_update_preserves_created_at():
    service, _, _ = make_service()

    created = await service.create(
        tenant_id="tenant-a",
        config=make_config(),
        server_id="server-a",
    )

    updated = await service.update(
        tenant_id="tenant-a",
        server_id="server-a",
        secret_references={"api_key": "secret://mcp/document"},
    )

    assert updated.created_at == created.created_at
    assert updated.updated_at >= created.updated_at


@pytest.mark.asyncio
async def test_delete_unregisters_runtime_before_deleting_durable_state():
    service, repository, manager = make_service()

    await service.create(
        tenant_id="tenant-a",
        config=make_config(),
        server_id="server-a",
    )

    deleted = await service.delete(
        tenant_id="tenant-a",
        server_id="server-a",
    )

    assert deleted.server_id == "server-a"
    assert repository.get("server-a") is None
    assert manager.list_servers() == []
    manager.unregister_server.assert_awaited_once_with("document-server")


@pytest.mark.asyncio
async def test_delete_does_not_allow_cross_tenant_deletion():
    service, repository, manager = make_service()

    await service.create(
        tenant_id="tenant-a",
        config=make_config(),
        desired_state=MCPServerDesiredState.DISABLED,
        server_id="server-a",
    )

    with pytest.raises(Exception, match="not found"):
        await service.delete(
            tenant_id="tenant-b",
            server_id="server-a",
        )

    assert repository.get("server-a") is not None
    assert manager.list_servers() == []


@pytest.mark.asyncio
async def test_reconcile_active_server_realizes_missing_runtime():
    service, repository, manager = make_service()

    server = await service.create(
        tenant_id="tenant-a",
        config=make_config(),
        desired_state=MCPServerDesiredState.DISABLED,
        server_id="server-a",
    )

    # Simulate an independently removed process-local runtime while the
    # durable desired state is changed back to ACTIVE.
    active = server.__class__(
        **{
            **server.__dict__,
            "desired_state": MCPServerDesiredState.ACTIVE,
            "updated_at": datetime.now(UTC),
        }
    )
    repository.update(active)

    await service.reconcile(
        tenant_id="tenant-a",
        server_id="server-a",
    )

    assert manager.list_servers() == ["document-server"]
    manager.register_server.assert_awaited_once()
    manager.connect_and_discover.assert_awaited_once_with("document-server")


@pytest.mark.asyncio
async def test_reconcile_disabled_server_removes_runtime():
    service, _, manager = make_service()

    await service.create(
        tenant_id="tenant-a",
        config=make_config(),
        server_id="server-a",
    )

    await service.set_desired_state(
        tenant_id="tenant-a",
        server_id="server-a",
        desired_state=MCPServerDesiredState.DISABLED,
    )

    # Re-register it to model stale process-local state.
    await manager.register_server(make_config())

    await service.reconcile(
        tenant_id="tenant-a",
        server_id="server-a",
    )

    assert manager.list_servers() == []
    assert manager.unregister_server.await_count == 2


@pytest.mark.asyncio
async def test_runtime_failure_removes_process_local_registration_but_keeps_active_desired_state():
    service, repository, manager = make_service()

    manager.connect_and_discover.side_effect = RuntimeError("MCP unavailable")

    with pytest.raises(RuntimeError, match="MCP unavailable"):
        await service.create(
            tenant_id="tenant-a",
            config=make_config(),
            server_id="server-a",
        )

    persisted = repository.get("server-a")

    assert persisted is not None
    assert persisted.desired_state is MCPServerDesiredState.ACTIVE
    assert manager.list_servers() == []
    manager.unregister_server.assert_awaited_once_with("document-server")


@pytest.mark.asyncio
async def test_reconcile_does_not_change_matching_runtime():
    service, _, manager = make_service()

    await service.create(
        tenant_id="tenant-a",
        config=make_config(),
        server_id="server-a",
    )

    manager.register_server.reset_mock()
    manager.connect_and_discover.reset_mock()
    manager.unregister_server.reset_mock()

    await service.reconcile(
        tenant_id="tenant-a",
        server_id="server-a",
    )

    manager.register_server.assert_not_awaited()
    manager.connect_and_discover.assert_not_awaited()
    manager.unregister_server.assert_not_awaited()
