from __future__ import annotations

from datetime import UTC, datetime

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.control_plane.dependencies import get_mcp_server_lifecycle_service
from app.control_plane.mcp_servers.exceptions import (
    MCPServerNotFoundError,
)
from app.control_plane.mcp_servers.models import (
    MCPServer,
    MCPServerDesiredState,
)
from app.control_plane.routes.mcp import router
from tools.mcp.config import MCPServerConfig
from tools.mcp.recovery import MCPRecoveryPolicy


def make_server(
    *,
    server_id: str = "server-1",
    tenant_id: str = "tenant-acme",
    name: str = "documents",
    desired_state: MCPServerDesiredState = MCPServerDesiredState.ACTIVE,
) -> MCPServer:
    now = datetime.now(UTC)

    return MCPServer(
        server_id=server_id,
        tenant_id=tenant_id,
        name=name,
        desired_state=desired_state,
        config=MCPServerConfig(
            name=name,
            transport="stdio",
            command="python",
            args=("server.py",),
            cwd="/opt/mcp",
            timeout=12.0,
            read_timeout=90.0,
            health_check_timeout=4.0,
            verify_ssl=False,
            recovery_policy=MCPRecoveryPolicy(
                max_attempts=4,
                initial_backoff=2.0,
                max_backoff=16.0,
                cooldown=30.0,
            ),
        ),
        secret_references={
            "authorization": "secret://mcp/documents/auth",
        },
        created_at=now,
        updated_at=now,
    )


class FakeLifecycleService:
    def __init__(self) -> None:
        self.servers = {
            "server-1": make_server(),
            "server-2": make_server(
                server_id="server-2",
                name="analytics",
                tenant_id="tenant-acme",
            ),
            "other-tenant": make_server(
                server_id="other-tenant",
                name="other",
                tenant_id="tenant-other",
            ),
        }

        self.get_calls: list[tuple[str, str]] = []
        self.list_calls: list[tuple[str, object, int]] = []
        self.create_calls: list[dict] = []
        self.update_calls: list[dict] = []
        self.reconcile_calls: list[tuple[str, str]] = []
        self.delete_calls: list[tuple[str, str]] = []

    def get(self, *, tenant_id: str, server_id: str) -> MCPServer:
        self.get_calls.append((tenant_id, server_id))

        server = self.servers.get(server_id)

        if server is None or server.tenant_id != tenant_id:
            raise MCPServerNotFoundError(f"MCP server not found for tenant: {server_id}")

        return server

    def list(
        self,
        *,
        tenant_id: str,
        desired_state=None,
        limit: int = 100,
    ) -> list[MCPServer]:
        self.list_calls.append((tenant_id, desired_state, limit))

        return [
            server
            for server in self.servers.values()
            if server.tenant_id == tenant_id
            and (desired_state is None or server.desired_state is desired_state)
        ][:limit]

    async def create(
        self,
        *,
        tenant_id: str,
        config,
        desired_state=MCPServerDesiredState.ACTIVE,
        secret_references=None,
        server_id=None,
    ) -> MCPServer:
        self.create_calls.append(
            {
                "tenant_id": tenant_id,
                "config": config,
                "desired_state": desired_state,
                "secret_references": secret_references,
            }
        )

        return make_server(
            server_id="created-server",
            tenant_id=tenant_id,
            name=config.name,
            desired_state=desired_state,
        )

    async def update(
        self,
        *,
        tenant_id: str,
        server_id: str,
        config=None,
        desired_state=None,
        secret_references=None,
    ) -> MCPServer:
        self.update_calls.append(
            {
                "tenant_id": tenant_id,
                "server_id": server_id,
                "config": config,
                "desired_state": desired_state,
                "secret_references": secret_references,
            }
        )

        existing = self.get(
            tenant_id=tenant_id,
            server_id=server_id,
        )

        return make_server(
            server_id=existing.server_id,
            tenant_id=existing.tenant_id,
            name=existing.name,
            desired_state=(desired_state if desired_state is not None else existing.desired_state),
        )

    async def reconcile(
        self,
        *,
        tenant_id: str,
        server_id: str,
    ) -> MCPServer:
        self.reconcile_calls.append((tenant_id, server_id))

        return self.get(
            tenant_id=tenant_id,
            server_id=server_id,
        )

    async def delete(
        self,
        *,
        tenant_id: str,
        server_id: str,
    ) -> MCPServer:
        self.delete_calls.append((tenant_id, server_id))

        return self.get(
            tenant_id=tenant_id,
            server_id=server_id,
        )


def build_client(
    service: FakeLifecycleService,
    *,
    principal: str | None = "api_key:test-owner",
    tenant_id: str | None = "tenant-acme",
) -> TestClient:
    app = FastAPI()
    app.include_router(router)

    app.dependency_overrides[get_mcp_server_lifecycle_service] = lambda: service

    class IdentityMiddleware:
        def __init__(self, inner_app):
            self.inner_app = inner_app

        async def __call__(self, scope, receive, send):
            state = scope.setdefault("state", {})

            if principal is not None:
                state["principal"] = principal

            if tenant_id is not None:
                state["tenant_id"] = tenant_id

            await self.inner_app(scope, receive, send)

    app.add_middleware(IdentityMiddleware)

    return TestClient(app)


CREATE_PAYLOAD = {
    "name": "documents",
    "transport": "stdio",
    "command": "python",
    "args": ["server.py"],
    "cwd": "/opt/mcp",
    "timeout": 12,
    "read_timeout": 90,
    "health_check_timeout": 4,
    "verify_ssl": False,
    "recovery_policy": {
        "max_attempts": 4,
        "initial_backoff": 2,
        "max_backoff": 16,
        "cooldown": 30,
    },
    "tool_capabilities": {},
    "desired_state": "active",
    "secret_references": {
        "authorization": "secret://mcp/documents/auth",
    },
}


def test_create_managed_server_uses_authenticated_tenant() -> None:
    service = FakeLifecycleService()
    client = build_client(service)

    response = client.post(
        "/api/v1/mcp/managed-servers",
        json=CREATE_PAYLOAD,
    )

    assert response.status_code == 201
    assert response.json()["tenant_id"] == "tenant-acme"
    assert service.create_calls[0]["tenant_id"] == "tenant-acme"
    assert service.create_calls[0]["config"].name == "documents"


def test_create_rejects_raw_environment_values() -> None:
    service = FakeLifecycleService()
    client = build_client(service)

    payload = {
        **CREATE_PAYLOAD,
        "env": {"TOKEN": "super-secret"},
    }

    response = client.post(
        "/api/v1/mcp/managed-servers",
        json=payload,
    )

    assert response.status_code == 422
    assert service.create_calls == []


def test_create_rejects_raw_http_headers() -> None:
    service = FakeLifecycleService()
    client = build_client(service)

    payload = {
        **CREATE_PAYLOAD,
        "headers": {
            "Authorization": "Bearer super-secret",
        },
    }

    response = client.post(
        "/api/v1/mcp/managed-servers",
        json=payload,
    )

    assert response.status_code == 422
    assert service.create_calls == []


def test_list_managed_servers_is_tenant_scoped() -> None:
    service = FakeLifecycleService()
    client = build_client(service)

    response = client.get(
        "/api/v1/mcp/managed-servers",
    )

    assert response.status_code == 200

    ids = {item["server_id"] for item in response.json()["servers"]}

    assert ids == {"server-1", "server-2"}
    assert service.list_calls == [
        ("tenant-acme", None, 100),
    ]


def test_list_managed_servers_supports_desired_state_filter() -> None:
    service = FakeLifecycleService()
    service.servers["server-2"] = make_server(
        server_id="server-2",
        name="analytics",
        desired_state=MCPServerDesiredState.DISABLED,
    )

    client = build_client(service)

    response = client.get(
        "/api/v1/mcp/managed-servers",
        params={"desired_state": "disabled"},
    )

    assert response.status_code == 200
    assert [item["server_id"] for item in response.json()["servers"]] == ["server-2"]


def test_list_rejects_invalid_desired_state() -> None:
    service = FakeLifecycleService()
    client = build_client(service)

    response = client.get(
        "/api/v1/mcp/managed-servers",
        params={"desired_state": "invalid"},
    )

    assert response.status_code == 422
    assert service.list_calls == []


def test_get_managed_server_uses_authenticated_tenant() -> None:
    service = FakeLifecycleService()
    client = build_client(service)

    response = client.get(
        "/api/v1/mcp/managed-servers/server-1",
    )

    assert response.status_code == 200
    assert response.json()["server_id"] == "server-1"
    assert service.get_calls == [
        ("tenant-acme", "server-1"),
    ]


def test_get_managed_server_blocks_cross_tenant_access() -> None:
    service = FakeLifecycleService()
    client = build_client(service)

    response = client.get(
        "/api/v1/mcp/managed-servers/other-tenant",
    )

    assert response.status_code == 404


def test_update_managed_server_uses_authenticated_tenant() -> None:
    service = FakeLifecycleService()
    client = build_client(service)

    response = client.patch(
        "/api/v1/mcp/managed-servers/server-1",
        json={
            "timeout": 25,
            "desired_state": "disabled",
        },
    )

    assert response.status_code == 200
    assert service.update_calls[0]["tenant_id"] == "tenant-acme"
    assert service.update_calls[0]["server_id"] == "server-1"
    assert service.update_calls[0]["config"].timeout == 25
    assert service.update_calls[0]["desired_state"] is MCPServerDesiredState.DISABLED


def test_update_rejects_name_and_transport_changes() -> None:
    service = FakeLifecycleService()
    client = build_client(service)

    response = client.patch(
        "/api/v1/mcp/managed-servers/server-1",
        json={
            "name": "attacker-name",
            "transport": "streamable-http",
        },
    )

    assert response.status_code == 422
    assert service.update_calls == []


def test_reconcile_uses_authenticated_tenant() -> None:
    service = FakeLifecycleService()
    client = build_client(service)

    response = client.post(
        "/api/v1/mcp/managed-servers/server-1/reconcile",
    )

    assert response.status_code == 200
    assert service.reconcile_calls == [
        ("tenant-acme", "server-1"),
    ]


def test_delete_uses_authenticated_tenant() -> None:
    service = FakeLifecycleService()
    client = build_client(service)

    response = client.delete(
        "/api/v1/mcp/managed-servers/server-1",
    )

    assert response.status_code == 200
    assert service.delete_calls == [
        ("tenant-acme", "server-1"),
    ]


def test_missing_identity_is_rejected() -> None:
    service = FakeLifecycleService()
    client = build_client(
        service,
        tenant_id=None,
    )

    response = client.get(
        "/api/v1/mcp/managed-servers",
    )

    assert response.status_code == 403
    assert service.list_calls == []


def test_request_body_cannot_override_tenant() -> None:
    service = FakeLifecycleService()
    client = build_client(service)

    payload = {
        **CREATE_PAYLOAD,
        "tenant_id": "tenant-attacker",
    }

    response = client.post(
        "/api/v1/mcp/managed-servers",
        json=payload,
    )

    assert response.status_code == 422
    assert service.create_calls == []
