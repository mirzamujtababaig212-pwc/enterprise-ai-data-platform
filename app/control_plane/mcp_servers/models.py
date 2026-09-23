from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from tools.mcp.config import MCPServerConfig


class MCPServerDesiredState(StrEnum):
    ACTIVE = "active"
    DISABLED = "disabled"


@dataclass(frozen=True)
class MCPServer:
    """
    Durable tenant-owned MCP server desired state.

    The MCPServerManager owns process-local runtime state. This model owns
    durable control-plane state describing what should exist.
    """

    server_id: str
    tenant_id: str
    name: str
    desired_state: MCPServerDesiredState
    config: MCPServerConfig
    secret_references: dict[str, str]
    created_at: datetime
    updated_at: datetime

    def __post_init__(self) -> None:
        if not self.server_id.strip():
            raise ValueError("MCP server id must not be empty.")

        if not self.tenant_id.strip():
            raise ValueError("MCP server tenant id must not be empty.")

        if not self.name.strip():
            raise ValueError("MCP server name must not be empty.")

        if self.config.name != self.name:
            raise ValueError("MCP server name must match the MCP server configuration name.")

        if not isinstance(self.desired_state, MCPServerDesiredState):
            raise TypeError("MCP server desired state must be an MCPServerDesiredState.")

        if any(
            not key.strip() or not value.strip() for key, value in self.secret_references.items()
        ):
            raise ValueError("MCP server secret references must contain nonblank keys and values.")
