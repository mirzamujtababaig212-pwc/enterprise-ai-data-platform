from __future__ import annotations

from datetime import datetime, timezone

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.control_plane.mcp_management_service import (
    MCPServerHealthInfo,
    MCPServerInfo,
)
from app.control_plane.routes.mcp import router
from app.control_plane.dependencies import get_mcp_management_service


class FakeMCPManagementService:
    def __init__(self) -> None:
        self.list_calls: list[str] = []
        self.health_calls: list[str] = []
        self.sync_calls: list[tuple[str, str]] = []

    async def list_servers(self, tenant_id: str) -> list[MCPServerInfo]:
        self.list_calls.append(tenant_id)

        return [
            MCPServerInfo(
                server_id="document-server",
                transport="streamable-http",
                connected=True,
            ),
        ]

    async def health(self, tenant_id: str) -> list[MCPServerHealthInfo]:
        self.health_calls.append(tenant_id)

        return [
            MCPServerHealthInfo(
                server_id="document-server",
                status="healthy",
                latency_ms=12.5,
                last_check=datetime.now(timezone.utc),
                error=None,
            ),
        ]

    async def sync(self, tenant_id: str, server_id: str):
        self.sync_calls.append((tenant_id, server_id))

        return [
            type(
                "FakeTool",
                (),
                {
                    "name": "documents.search",
                    "description": "Search enterprise documents.",
                    "input_schema": {"type": "object"},
                    "metadata": {
                        "source": "mcp",
                        "mcp_server": server_id,
                    },
                    "enabled": True,
                },
            )()
        ]


def build_client(
    service: FakeMCPManagementService,
    *,
    principal: str | None = "api_key:test-owner",
    tenant_id: str | None = "tenant-acme",
) -> TestClient:
    app = FastAPI()
    app.include_router(router)

    app.dependency_overrides[get_mcp_management_service] = lambda: service

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


def test_list_mcp_servers_uses_authenticated_tenant() -> None:
    service = FakeMCPManagementService()
    client = build_client(service)

    response = client.get("/api/v1/mcp/servers")

    assert response.status_code == 200
    assert response.json() == {
        "servers": [
            {
                "server_id": "document-server",
                "transport": "streamable-http",
                "connected": True,
            }
        ]
    }
    assert service.list_calls == ["tenant-acme"]


def test_mcp_health_uses_authenticated_tenant() -> None:
    service = FakeMCPManagementService()
    client = build_client(service)

    response = client.get("/api/v1/mcp/servers/health")

    assert response.status_code == 200
    body = response.json()

    assert len(body["servers"]) == 1
    assert body["servers"][0]["server_id"] == "document-server"
    assert body["servers"][0]["status"] == "healthy"
    assert body["servers"][0]["latency_ms"] == 12.5
    assert body["servers"][0]["error"] is None
    assert service.health_calls == ["tenant-acme"]


def test_mcp_sync_uses_authenticated_tenant() -> None:
    service = FakeMCPManagementService()
    client = build_client(service)

    response = client.post(
        "/api/v1/mcp/servers/document-server/sync",
    )

    assert response.status_code == 200
    assert response.json() == {
        "server_id": "document-server",
        "tools": [
            {
                "name": "documents.search",
                "description": "Search enterprise documents.",
                "input_schema": {"type": "object"},
                "metadata": {
                    "source": "mcp",
                    "mcp_server": "document-server",
                },
                "enabled": True,
            }
        ],
    }
    assert service.sync_calls == [
        ("tenant-acme", "document-server"),
    ]


def test_mcp_routes_reject_missing_tenant_context() -> None:
    service = FakeMCPManagementService()
    client = build_client(
        service,
        tenant_id=None,
    )

    response = client.get("/api/v1/mcp/servers")

    assert response.status_code == 403
    assert service.list_calls == []


def test_mcp_routes_reject_missing_principal_context() -> None:
    service = FakeMCPManagementService()
    client = build_client(
        service,
        principal=None,
    )

    response = client.get("/api/v1/mcp/servers")

    assert response.status_code == 403
    assert service.list_calls == []


def test_mcp_sync_does_not_accept_tenant_from_request_body() -> None:
    service = FakeMCPManagementService()
    client = build_client(service, tenant_id="tenant-acme")

    response = client.post(
        "/api/v1/mcp/servers/document-server/sync",
        json={"tenant_id": "tenant-attacker"},
    )

    assert response.status_code == 200
    assert service.sync_calls == [
        ("tenant-acme", "document-server"),
    ]
