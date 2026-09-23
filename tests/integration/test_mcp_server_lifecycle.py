"""PostgreSQL integration tests for the MCP server lifecycle API."""

from __future__ import annotations

import os
from dataclasses import dataclass

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import delete

from app.control_plane.dependencies import get_mcp_server_lifecycle_service
from app.control_plane.mcp_servers.lifecycle_service import (
    MCPServerLifecycleService,
)
from app.control_plane.mcp_servers.postgres_repository import (
    PostgreSQLMCPServerRepository,
)
from app.control_plane.persistence.database import SessionLocal
from app.control_plane.persistence.models import MCPServerRecord
from app.control_plane.routes.mcp import router
from tools.mcp.config import MCPServerConfig

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_POSTGRES_INTEGRATION") != "1",
    reason="Set RUN_POSTGRES_INTEGRATION=1 to run the PostgreSQL integration test",
)


@dataclass
class FakeRuntimeManager:
    """Deterministic process-local MCP runtime for API integration tests."""

    registered: dict[str, MCPServerConfig]

    def __init__(self) -> None:
        self.registered = {}

    async def register_server(self, config: MCPServerConfig) -> None:
        if config.name in self.registered:
            raise ValueError(f"server already registered: {config.name}")
        self.registered[config.name] = config

    async def connect_and_discover(self, name: str) -> None:
        if name not in self.registered:
            raise RuntimeError(f"server is not registered: {name}")

    async def unregister_server(self, name: str) -> None:
        self.registered.pop(name, None)

    def get_config(self, name: str) -> MCPServerConfig:
        return self.registered[name]

    def list_servers(self) -> list[str]:
        return list(self.registered)


def make_config(
    name: str = "integration-lifecycle-server",
    *,
    command: str = "integration-mcp",
) -> MCPServerConfig:
    return MCPServerConfig(
        name=name,
        transport="stdio",
        command=command,
    )


def build_client(
    service: MCPServerLifecycleService,
    *,
    tenant_id: str = "integration-tenant-a",
    principal: str = "api_key:integration-owner",
) -> TestClient:
    app = FastAPI()
    app.include_router(router)

    app.dependency_overrides[get_mcp_server_lifecycle_service] = lambda: service

    class IdentityMiddleware:
        def __init__(self, inner_app):
            self.inner_app = inner_app

        async def __call__(self, scope, receive, send):
            state = scope.setdefault("state", {})
            state["tenant_id"] = tenant_id
            state["principal"] = principal
            await self.inner_app(scope, receive, send)

    app.add_middleware(IdentityMiddleware)

    return TestClient(app)


def delete_server(server_id: str) -> None:
    with SessionLocal() as session:
        session.execute(
            delete(MCPServerRecord).where(
                MCPServerRecord.server_id == server_id,
            )
        )
        session.commit()


def test_postgres_mcp_lifecycle_api_round_trip_and_tenant_isolation() -> None:
    server_id = "integration-mcp-lifecycle-api"
    server_name = "integration-lifecycle-server"

    runtime_manager = FakeRuntimeManager()

    with SessionLocal() as session:
        repository = PostgreSQLMCPServerRepository(session)
        service = MCPServerLifecycleService(
            repository=repository,
            manager=runtime_manager,
        )

        client = build_client(service)

        try:
            create_response = client.post(
                "/api/v1/mcp/managed-servers",
                json={
                    "name": server_name,
                    "transport": "stdio",
                    "command": "integration-mcp",
                    "desired_state": "active",
                    "secret_references": {
                        "authorization": "secret://mcp/integration/auth",
                    },
                },
            )

            assert create_response.status_code == 201
            created = create_response.json()
            server_id = created["server_id"]

            assert server_id
            assert created["tenant_id"] == "integration-tenant-a"
            assert created["name"] == server_name
            assert created["desired_state"] == "active"
            assert created["secret_references"] == {
                "authorization": "secret://mcp/integration/auth",
            }

            assert runtime_manager.list_servers() == [server_name]

            get_response = client.get(
                f"/api/v1/mcp/managed-servers/{server_id}",
            )

            assert get_response.status_code == 200
            assert get_response.json()["server_id"] == server_id

            disable_response = client.patch(
                f"/api/v1/mcp/managed-servers/{server_id}",
                json={"desired_state": "disabled"},
            )

            assert disable_response.status_code == 200
            assert disable_response.json()["desired_state"] == "disabled"
            assert runtime_manager.list_servers() == []

            with SessionLocal() as verification_session:
                record = verification_session.get(
                    MCPServerRecord,
                    server_id,
                )

                assert record is not None
                assert record.tenant_id == "integration-tenant-a"
                assert record.name == server_name
                assert record.desired_state == "disabled"
                assert record.secret_references == {
                    "authorization": "secret://mcp/integration/auth",
                }

            activate_response = client.patch(
                f"/api/v1/mcp/managed-servers/{server_id}",
                json={"desired_state": "active"},
            )

            assert activate_response.status_code == 200
            assert activate_response.json()["desired_state"] == "active"
            assert runtime_manager.list_servers() == [server_name]

            await_reconcile_response = client.post(
                f"/api/v1/mcp/managed-servers/{server_id}/reconcile",
            )

            assert await_reconcile_response.status_code == 200
            assert await_reconcile_response.json()["server_id"] == server_id

            delete_response = client.delete(
                f"/api/v1/mcp/managed-servers/{server_id}",
            )

            assert delete_response.status_code == 200
            assert delete_response.json()["server_id"] == server_id
            assert runtime_manager.list_servers() == []

            missing_response = client.get(
                f"/api/v1/mcp/managed-servers/{server_id}",
            )

            assert missing_response.status_code == 404

            with SessionLocal() as verification_session:
                assert (
                    verification_session.get(
                        MCPServerRecord,
                        server_id,
                    )
                    is None
                )

        finally:
            delete_server(server_id)


def test_postgres_mcp_lifecycle_api_blocks_cross_tenant_access() -> None:
    server_id = "integration-mcp-lifecycle-tenant"
    server_name = "integration-lifecycle-tenant"

    runtime_manager = FakeRuntimeManager()

    with SessionLocal() as session:
        repository = PostgreSQLMCPServerRepository(session)
        service = MCPServerLifecycleService(
            repository=repository,
            manager=runtime_manager,
        )

        owner_client = build_client(
            service,
            tenant_id="integration-tenant-owner",
            principal="api_key:integration-owner",
        )

        other_tenant_client = build_client(
            service,
            tenant_id="integration-tenant-other",
            principal="api_key:integration-other",
        )

        try:
            create_response = owner_client.post(
                "/api/v1/mcp/managed-servers",
                json={
                    "name": server_name,
                    "transport": "stdio",
                    "command": "integration-mcp",
                    "desired_state": "disabled",
                },
            )

            assert create_response.status_code == 201
            created = create_response.json()
            server_id = created["server_id"]

            get_response = other_tenant_client.get(
                f"/api/v1/mcp/managed-servers/{server_id}",
            )

            assert get_response.status_code == 404

            update_response = other_tenant_client.patch(
                f"/api/v1/mcp/managed-servers/{server_id}",
                json={"desired_state": "active"},
            )

            assert update_response.status_code == 404

            delete_response = other_tenant_client.delete(
                f"/api/v1/mcp/managed-servers/{server_id}",
            )

            assert delete_response.status_code == 404

            owner_get_response = owner_client.get(
                f"/api/v1/mcp/managed-servers/{server_id}",
            )

            assert owner_get_response.status_code == 200
            assert owner_get_response.json()["tenant_id"] == "integration-tenant-owner"

        finally:
            delete_server(server_id)
