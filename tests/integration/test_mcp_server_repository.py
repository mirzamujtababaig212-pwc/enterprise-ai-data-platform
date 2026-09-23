"""PostgreSQL integration tests for durable MCP server state."""

from __future__ import annotations

import os
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import delete, select

from app.control_plane.mcp_servers.exceptions import (
    MCPServerAlreadyExistsError,
)
from app.control_plane.mcp_servers.models import (
    MCPServer,
    MCPServerDesiredState,
)
from app.control_plane.mcp_servers.postgres_repository import (
    PostgreSQLMCPServerRepository,
)
from app.control_plane.persistence.database import SessionLocal
from app.control_plane.persistence.models import MCPServerRecord
from tests.control_plane.test_mcp_server_repositories import make_config

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_POSTGRES_INTEGRATION") != "1",
    reason="Set RUN_POSTGRES_INTEGRATION=1 to run the PostgreSQL integration test",
)


def make_server(
    server_id: str = "integration-mcp-server",
    tenant_id: str = "tenant-a",
    name: str = "integration-documents",
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
            "http.authorization": "secret://mcp/integration/authorization",
        },
        created_at=now,
        updated_at=now,
    )


def _delete_server(server_id: str) -> None:
    with SessionLocal() as session:
        session.execute(
            delete(MCPServerRecord).where(
                MCPServerRecord.server_id == server_id,
            )
        )
        session.commit()


def test_postgres_repository_round_trip_preserves_durable_mcp_state() -> None:
    server = make_server()

    try:
        with SessionLocal() as session:
            repository = PostgreSQLMCPServerRepository(session)

            created = repository.create(server)
            assert created == server

        with SessionLocal() as session:
            repository = PostgreSQLMCPServerRepository(session)

            restored = repository.get(server.server_id)

        assert restored == server
        assert restored is not server
        assert restored.config == server.config
        assert restored.config.env == {}
        assert restored.config.headers == {}
        assert restored.secret_references == server.secret_references

    finally:
        _delete_server(server.server_id)


def test_postgres_repository_scopes_reads_by_tenant() -> None:
    server = make_server(
        server_id="integration-mcp-tenant-scope",
        name="integration-tenant-scope",
        tenant_id="tenant-a",
    )

    try:
        with SessionLocal() as session:
            PostgreSQLMCPServerRepository(session).create(server)

        with SessionLocal() as session:
            repository = PostgreSQLMCPServerRepository(session)

            assert (
                repository.get_for_tenant(
                    server.server_id,
                    "tenant-a",
                )
                == server
            )

            assert (
                repository.get_for_tenant(
                    server.server_id,
                    "tenant-b",
                )
                is None
            )

    finally:
        _delete_server(server.server_id)


def test_postgres_repository_lists_by_tenant_and_desired_state() -> None:
    active = make_server(
        server_id="integration-mcp-active",
        name="integration-active",
        tenant_id="tenant-a",
        desired_state=MCPServerDesiredState.ACTIVE,
    )
    disabled = make_server(
        server_id="integration-mcp-disabled",
        name="integration-disabled",
        tenant_id="tenant-a",
        desired_state=MCPServerDesiredState.DISABLED,
    )
    other_tenant = make_server(
        server_id="integration-mcp-other-tenant",
        name="integration-other-tenant",
        tenant_id="tenant-b",
        desired_state=MCPServerDesiredState.ACTIVE,
    )

    servers = [active, disabled, other_tenant]

    try:
        with SessionLocal() as session:
            repository = PostgreSQLMCPServerRepository(session)

            for server in servers:
                repository.create(server)

        with SessionLocal() as session:
            repository = PostgreSQLMCPServerRepository(session)

            assert repository.list(tenant_id="tenant-a") == [
                disabled,
                active,
            ]

            assert repository.list(
                tenant_id="tenant-a",
                desired_state=MCPServerDesiredState.ACTIVE.value,
            ) == [active]

            assert repository.list(
                tenant_id="tenant-a",
                desired_state=MCPServerDesiredState.DISABLED.value,
            ) == [disabled]

            assert repository.list(tenant_id="tenant-b") == [other_tenant]

    finally:
        for server in servers:
            _delete_server(server.server_id)


def test_postgres_repository_update_persists_desired_state_and_configuration() -> None:
    server = make_server(
        server_id="integration-mcp-update",
        name="integration-update",
    )

    try:
        with SessionLocal() as session:
            repository = PostgreSQLMCPServerRepository(session)
            repository.create(server)

        updated = replace(
            server,
            desired_state=MCPServerDesiredState.DISABLED,
            config=make_config(
                "integration-update",
            ),
            updated_at=server.updated_at + timedelta(minutes=1),
        )

        with SessionLocal() as session:
            repository = PostgreSQLMCPServerRepository(session)

            assert repository.update(updated) == updated

        with SessionLocal() as session:
            repository = PostgreSQLMCPServerRepository(session)

            restored = repository.get(server.server_id)

        assert restored == updated
        assert restored.desired_state is MCPServerDesiredState.DISABLED

    finally:
        _delete_server(server.server_id)


def test_postgres_repository_enforces_globally_unique_server_name() -> None:
    first = make_server(
        server_id="integration-mcp-name-one",
        name="integration-global-name",
        tenant_id="tenant-a",
    )
    second = make_server(
        server_id="integration-mcp-name-two",
        name="integration-global-name",
        tenant_id="tenant-b",
    )

    try:
        with SessionLocal() as session:
            repository = PostgreSQLMCPServerRepository(session)

            repository.create(first)

        with SessionLocal() as session:
            repository = PostgreSQLMCPServerRepository(session)

            with pytest.raises(
                MCPServerAlreadyExistsError,
                match="integration-mcp-name-two",
            ):
                repository.create(second)

    finally:
        _delete_server(first.server_id)
        _delete_server(second.server_id)


def test_postgres_repository_delete_requires_matching_tenant() -> None:
    server = make_server(
        server_id="integration-mcp-delete",
        name="integration-delete",
        tenant_id="tenant-a",
    )

    try:
        with SessionLocal() as session:
            PostgreSQLMCPServerRepository(session).create(server)

        with SessionLocal() as session:
            repository = PostgreSQLMCPServerRepository(session)

            assert (
                repository.delete(
                    server.server_id,
                    tenant_id="tenant-b",
                )
                is None
            )

        with SessionLocal() as session:
            repository = PostgreSQLMCPServerRepository(session)

            assert (
                repository.delete(
                    server.server_id,
                    tenant_id="tenant-a",
                )
                == server
            )

        with SessionLocal() as session:
            repository = PostgreSQLMCPServerRepository(session)

            assert repository.get(server.server_id) is None

    finally:
        _delete_server(server.server_id)


def test_postgres_repository_does_not_persist_secret_bearing_config_fields() -> None:
    server = make_server(
        server_id="integration-mcp-secret-boundary",
        name="integration-secret-boundary",
    )

    try:
        with SessionLocal() as session:
            repository = PostgreSQLMCPServerRepository(session)
            repository.create(server)

        with SessionLocal() as session:
            record = session.scalar(
                select(MCPServerRecord).where(
                    MCPServerRecord.server_id == server.server_id,
                )
            )

            assert record is not None
            assert "env" not in record.configuration
            assert "headers" not in record.configuration
            assert record.secret_references == server.secret_references

    finally:
        _delete_server(server.server_id)
