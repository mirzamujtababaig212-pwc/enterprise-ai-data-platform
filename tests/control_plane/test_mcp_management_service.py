from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from ai_platform.agents.policy import (
    PolicyViolationError,
    TenantPolicy,
    TenantPolicyEngine,
)
from app.control_plane.mcp_management_service import (
    MCPManagementService,
)


class FakeMCPManager:
    def __init__(self) -> None:
        self.servers = {
            "document-server": SimpleNamespace(
                transport="streamable-http",
                connected=True,
            ),
            "vehicle-server": SimpleNamespace(
                transport="stdio",
                connected=False,
            ),
        }
        self.health_calls: list[str] = []
        self.sync_calls: list[str] = []

    def list_servers(self) -> list[str]:
        return list(self.servers)

    def get_config(self, name: str):
        server = self.servers[name]
        return SimpleNamespace(
            name=name,
            transport=server.transport,
            url=("http://document-server/mcp" if name == "document-server" else None),
        )

    def is_connected(self, name: str) -> bool:
        return self.servers[name].connected

    async def check_health(self, name: str):
        self.health_calls.append(name)
        return SimpleNamespace(
            server_name=name,
            status=SimpleNamespace(value="healthy"),
            latency_ms=12.5,
            last_check=datetime.now(timezone.utc),
            error=None,
        )

    async def discover_server(self, name: str):
        self.sync_calls.append(name)
        return [
            SimpleNamespace(
                name="documents.search",
                description="Search enterprise documents.",
                input_schema={"type": "object"},
                metadata={
                    "source": "mcp",
                    "mcp_server": name,
                },
                enabled=True,
            )
        ]


def build_service(
    *,
    allowed_mcp_servers: frozenset[str] = frozenset(),
) -> tuple[MCPManagementService, FakeMCPManager]:
    manager = FakeMCPManager()

    policy_engine = TenantPolicyEngine()
    policy_engine.register_policy(
        TenantPolicy(
            tenant_id="tenant-acme",
            allowed_mcp_servers=allowed_mcp_servers,
        )
    )

    return (
        MCPManagementService(
            manager=manager,
            tenant_policy_engine=policy_engine,
        ),
        manager,
    )


@pytest.mark.asyncio
async def test_list_servers_returns_all_servers_when_policy_has_no_mcp_restriction():
    service, _ = build_service()

    servers = await service.list_servers("tenant-acme")

    assert [server.server_id for server in servers] == [
        "document-server",
        "vehicle-server",
    ]


@pytest.mark.asyncio
async def test_list_servers_filters_to_tenant_allowed_servers():
    service, _ = build_service(
        allowed_mcp_servers=frozenset({"document-server"}),
    )

    servers = await service.list_servers("tenant-acme")

    assert [server.server_id for server in servers] == ["document-server"]


@pytest.mark.asyncio
async def test_health_filters_to_tenant_allowed_servers():
    service, manager = build_service(
        allowed_mcp_servers=frozenset({"document-server"}),
    )

    health = await service.health("tenant-acme")

    assert [item.server_id for item in health] == ["document-server"]
    assert manager.health_calls == ["document-server"]
    assert health[0].status == "healthy"


@pytest.mark.asyncio
async def test_sync_rejects_server_not_allowed_for_tenant():
    service, manager = build_service(
        allowed_mcp_servers=frozenset({"document-server"}),
    )

    with pytest.raises(PolicyViolationError, match="not authorized"):
        await service.sync("tenant-acme", "vehicle-server")

    assert manager.sync_calls == []


@pytest.mark.asyncio
async def test_restricted_tenant_cannot_enumerate_unknown_server():
    service, manager = build_service(
        allowed_mcp_servers=frozenset({"document-server"}),
    )

    with pytest.raises(PolicyViolationError, match="not authorized"):
        await service.sync("tenant-acme", "missing-server")

    assert manager.sync_calls == []


@pytest.mark.asyncio
async def test_sync_discovers_tools_for_authorized_server():
    service, manager = build_service(
        allowed_mcp_servers=frozenset({"document-server"}),
    )

    tools = await service.sync("tenant-acme", "document-server")

    assert manager.sync_calls == ["document-server"]
    assert tools[0].name == "documents.search"
    assert tools[0].metadata["mcp_server"] == "document-server"


@pytest.mark.asyncio
async def test_unknown_server_is_not_exposed():
    service, _ = build_service()

    with pytest.raises(KeyError):
        await service.sync("tenant-acme", "missing-server")
