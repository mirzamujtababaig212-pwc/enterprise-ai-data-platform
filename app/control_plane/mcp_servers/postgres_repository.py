from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.control_plane.mcp_servers.exceptions import (
    MCPServerAlreadyExistsError,
    MCPServerNotFoundError,
)
from app.control_plane.mcp_servers.models import (
    MCPServer,
    MCPServerDesiredState,
)
from app.control_plane.mcp_servers.serialization import (
    deserialize_mcp_server_config,
    serialize_mcp_server_config,
)
from app.control_plane.persistence.models import MCPServerRecord


class PostgreSQLMCPServerRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def create(
        self,
        server: MCPServer,
        *,
        commit: bool = True,
    ) -> MCPServer:
        existing = self._session.scalar(
            select(MCPServerRecord).where(MCPServerRecord.server_id == server.server_id)
        )

        if existing is not None:
            raise MCPServerAlreadyExistsError(f"MCP server already exists: {server.server_id}")

        record = MCPServerRecord(
            server_id=server.server_id,
            tenant_id=server.tenant_id,
            name=server.name,
            transport=server.config.transport,
            desired_state=server.desired_state.value,
            configuration=serialize_mcp_server_config(server.config),
            secret_references=dict(server.secret_references),
            created_at=server.created_at,
            updated_at=server.updated_at,
        )

        self._session.add(record)

        try:
            self._session.flush()

            if commit:
                self._session.commit()
        except IntegrityError as exc:
            if commit:
                self._session.rollback()

            raise MCPServerAlreadyExistsError(
                f"MCP server already exists: {server.server_id}"
            ) from exc
        except Exception:
            if commit:
                self._session.rollback()
            raise

        return server

    def get(self, server_id: str) -> MCPServer | None:
        record = self._session.scalar(
            select(MCPServerRecord).where(MCPServerRecord.server_id == server_id)
        )

        if record is None:
            return None

        return self._to_domain(record)

    def get_for_tenant(
        self,
        server_id: str,
        tenant_id: str,
    ) -> MCPServer | None:
        record = self._session.scalar(
            select(MCPServerRecord).where(
                MCPServerRecord.server_id == server_id,
                MCPServerRecord.tenant_id == tenant_id,
            )
        )

        if record is None:
            return None

        return self._to_domain(record)

    def get_by_name(self, name: str) -> MCPServer | None:
        record = self._session.scalar(
            select(MCPServerRecord).where(
                MCPServerRecord.name == name,
            )
        )

        if record is None:
            return None

        return self._to_domain(record)

    def list(
        self,
        *,
        tenant_id: str | None = None,
        desired_state: str | None = None,
        limit: int = 100,
    ) -> list[MCPServer]:
        if limit <= 0:
            raise ValueError("limit must be greater than zero.")

        statement = select(MCPServerRecord)

        if tenant_id is not None:
            statement = statement.where(MCPServerRecord.tenant_id == tenant_id)

        if desired_state is not None:
            statement = statement.where(MCPServerRecord.desired_state == desired_state)

        statement = statement.order_by(
            MCPServerRecord.created_at.desc(),
            MCPServerRecord.server_id.desc(),
        ).limit(limit)

        return [self._to_domain(record) for record in self._session.scalars(statement).all()]

    def update(
        self,
        server: MCPServer,
        *,
        commit: bool = True,
    ) -> MCPServer:
        record = self._session.scalar(
            select(MCPServerRecord).where(MCPServerRecord.server_id == server.server_id)
        )

        if record is None:
            raise MCPServerNotFoundError(f"MCP server not found: {server.server_id}")

        record.tenant_id = server.tenant_id
        record.name = server.name
        record.transport = server.config.transport
        record.desired_state = server.desired_state.value
        record.configuration = serialize_mcp_server_config(server.config)
        record.secret_references = dict(server.secret_references)
        record.created_at = server.created_at
        record.updated_at = server.updated_at

        try:
            self._session.flush()

            if commit:
                self._session.commit()
        except IntegrityError as exc:
            if commit:
                self._session.rollback()

            raise MCPServerAlreadyExistsError(
                f"MCP server name already exists: {server.name}"
            ) from exc
        except Exception:
            if commit:
                self._session.rollback()
            raise

        return server

    def delete(
        self,
        server_id: str,
        *,
        tenant_id: str,
        commit: bool = True,
    ) -> MCPServer | None:
        record = self._session.scalar(
            select(MCPServerRecord).where(
                MCPServerRecord.server_id == server_id,
                MCPServerRecord.tenant_id == tenant_id,
            )
        )

        if record is None:
            return None

        server = self._to_domain(record)
        self._session.delete(record)

        try:
            self._session.flush()

            if commit:
                self._session.commit()
        except Exception:
            if commit:
                self._session.rollback()
            raise

        return server

    @staticmethod
    def _to_domain(record: MCPServerRecord) -> MCPServer:
        configuration = dict(record.configuration)

        return MCPServer(
            server_id=record.server_id,
            tenant_id=record.tenant_id,
            name=record.name,
            desired_state=MCPServerDesiredState(record.desired_state),
            config=deserialize_mcp_server_config(configuration),
            secret_references=dict(record.secret_references or {}),
            created_at=record.created_at,
            updated_at=record.updated_at,
        )
