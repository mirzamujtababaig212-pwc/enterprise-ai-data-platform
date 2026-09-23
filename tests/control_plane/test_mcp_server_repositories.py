from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from app.control_plane.mcp_servers.exceptions import (
    MCPServerAlreadyExistsError,
    MCPServerNotFoundError,
)
from app.control_plane.mcp_servers.in_memory import (
    InMemoryMCPServerRepository,
)
from app.control_plane.mcp_servers.models import (
    MCPServer,
    MCPServerDesiredState,
)
from app.control_plane.mcp_servers.serialization import (
    deserialize_mcp_server_config,
    serialize_mcp_server_config,
)
from tools.mcp.config import MCPServerConfig, MCPToolCapability
from tools.mcp.recovery import MCPRecoveryPolicy


def make_config(
    name: str = "documents",
    *,
    env: dict[str, str] | None = None,
    headers: dict[str, str] | None = None,
) -> MCPServerConfig:
    return MCPServerConfig(
        name=name,
        transport="stdio",
        command="python",
        args=("server.py", "--port", "9000"),
        env=env or {},
        headers=headers or {},
        cwd="/opt/mcp/documents",
        timeout=12.5,
        read_timeout=120.0,
        health_check_timeout=7.5,
        verify_ssl=False,
        recovery_policy=MCPRecoveryPolicy(
            max_attempts=5,
            initial_backoff=2.0,
            max_backoff=45.0,
            cooldown=90.0,
        ),
        tool_capabilities={
            "documents.search": MCPToolCapability(
                capability="search",
                risk_tier="low",
                side_effect=False,
                permission_scope="documents:read",
            ),
            "documents.delete": MCPToolCapability(
                capability="delete",
                risk_tier="high",
                side_effect=True,
                permission_scope="documents:write",
            ),
        },
    )


def make_server(
    server_id: str = "mcp-documents-1",
    tenant_id: str = "tenant-a",
    name: str = "documents",
    *,
    desired_state: MCPServerDesiredState = MCPServerDesiredState.ACTIVE,
) -> MCPServer:
    now = datetime.now(UTC)

    return MCPServer(
        server_id=server_id,
        tenant_id=tenant_id,
        name=name,
        desired_state=desired_state,
        config=make_config(name),
        secret_references={
            "http.authorization": "secret://mcp/documents/authorization",
        },
        created_at=now,
        updated_at=now,
    )


def test_config_serialization_round_trip_preserves_non_secret_configuration() -> None:
    config = make_config()

    serialized = serialize_mcp_server_config(config)

    assert serialized == {
        "name": "documents",
        "transport": "stdio",
        "command": "python",
        "args": ["server.py", "--port", "9000"],
        "cwd": "/opt/mcp/documents",
        "url": None,
        "timeout": 12.5,
        "read_timeout": 120.0,
        "health_check_timeout": 7.5,
        "verify_ssl": False,
        "recovery_policy": {
            "max_attempts": 5,
            "initial_backoff": 2.0,
            "max_backoff": 45.0,
            "cooldown": 90.0,
        },
        "tool_capabilities": {
            "documents.search": {
                "capability": "search",
                "risk_tier": "low",
                "side_effect": False,
                "permission_scope": "documents:read",
            },
            "documents.delete": {
                "capability": "delete",
                "risk_tier": "high",
                "side_effect": True,
                "permission_scope": "documents:write",
            },
        },
    }

    restored = deserialize_mcp_server_config(serialized)

    assert restored == config


def test_config_serialization_rejects_environment_values() -> None:
    config = make_config(env={"API_TOKEN": "super-secret"})

    with pytest.raises(
        ValueError,
        match="environment values must not be persisted",
    ):
        serialize_mcp_server_config(config)


def test_config_serialization_rejects_http_headers() -> None:
    config = make_config(headers={"Authorization": "Bearer super-secret"})

    with pytest.raises(
        ValueError,
        match="HTTP headers must not be persisted",
    ):
        serialize_mcp_server_config(config)


def test_deserializer_does_not_reintroduce_secret_fields() -> None:
    serialized = serialize_mcp_server_config(make_config())

    serialized["env"] = {"API_TOKEN": "should-not-become-runtime-secret"}
    serialized["headers"] = {"Authorization": "Bearer should-not-become-runtime-secret"}

    restored = deserialize_mcp_server_config(serialized)

    assert restored.env == {}
    assert restored.headers == {}


def test_in_memory_repository_enforces_unique_server_id() -> None:
    repository = InMemoryMCPServerRepository()
    server = make_server()

    repository.create(server)

    with pytest.raises(
        MCPServerAlreadyExistsError,
        match="mcp-documents-1",
    ):
        repository.create(server)


def test_in_memory_repository_enforces_globally_unique_name() -> None:
    repository = InMemoryMCPServerRepository()

    repository.create(make_server())

    with pytest.raises(
        MCPServerAlreadyExistsError,
        match="documents",
    ):
        repository.create(
            make_server(
                server_id="mcp-documents-2",
                tenant_id="tenant-b",
            )
        )


def test_in_memory_repository_get_for_tenant_prevents_cross_tenant_access() -> None:
    repository = InMemoryMCPServerRepository()
    server = make_server(tenant_id="tenant-a")

    repository.create(server)

    assert repository.get_for_tenant(server.server_id, "tenant-a") == server
    assert repository.get_for_tenant(server.server_id, "tenant-b") is None


def test_in_memory_repository_list_is_tenant_scoped() -> None:
    repository = InMemoryMCPServerRepository()

    tenant_a = make_server(
        server_id="server-a",
        tenant_id="tenant-a",
        name="server-a",
    )
    tenant_b = make_server(
        server_id="server-b",
        tenant_id="tenant-b",
        name="server-b",
    )

    repository.create(tenant_a)
    repository.create(tenant_b)

    assert repository.list(tenant_id="tenant-a") == [tenant_a]
    assert repository.list(tenant_id="tenant-b") == [tenant_b]


def test_in_memory_repository_filters_desired_state() -> None:
    repository = InMemoryMCPServerRepository()

    active = make_server(
        server_id="server-active",
        name="server-active",
        desired_state=MCPServerDesiredState.ACTIVE,
    )
    disabled = make_server(
        server_id="server-disabled",
        name="server-disabled",
        desired_state=MCPServerDesiredState.DISABLED,
    )

    repository.create(active)
    repository.create(disabled)

    assert repository.list(desired_state="active") == [active]
    assert repository.list(desired_state="disabled") == [disabled]


def test_in_memory_repository_update_replaces_existing_record() -> None:
    repository = InMemoryMCPServerRepository()
    original = make_server()

    repository.create(original)

    updated = replace(
        original,
        desired_state=MCPServerDesiredState.DISABLED,
        updated_at=original.updated_at + timedelta(minutes=1),
    )

    assert repository.update(updated) == updated
    assert repository.get(original.server_id) == updated


def test_in_memory_repository_update_rejects_missing_record() -> None:
    repository = InMemoryMCPServerRepository()

    with pytest.raises(
        MCPServerNotFoundError,
        match="missing",
    ):
        repository.update(
            make_server(
                server_id="missing",
                name="missing",
            )
        )


def test_in_memory_repository_update_rejects_name_collision() -> None:
    repository = InMemoryMCPServerRepository()

    first = make_server(
        server_id="server-1",
        name="server-1",
    )
    second = make_server(
        server_id="server-2",
        name="server-2",
    )

    repository.create(first)
    repository.create(second)

    with pytest.raises(
        MCPServerAlreadyExistsError,
        match="server-2",
    ):
        repository.update(
            replace(
                first,
                name=second.name,
                config=make_config(second.name),
            )
        )


def test_in_memory_repository_delete_requires_matching_tenant() -> None:
    repository = InMemoryMCPServerRepository()
    server = make_server(tenant_id="tenant-a")

    repository.create(server)

    assert (
        repository.delete(
            server.server_id,
            tenant_id="tenant-b",
        )
        is None
    )

    assert repository.get(server.server_id) == server

    assert (
        repository.delete(
            server.server_id,
            tenant_id="tenant-a",
        )
        == server
    )

    assert repository.get(server.server_id) is None


def test_in_memory_repository_rejects_non_positive_limit() -> None:
    repository = InMemoryMCPServerRepository()

    with pytest.raises(ValueError, match="greater than zero"):
        repository.list(limit=0)
