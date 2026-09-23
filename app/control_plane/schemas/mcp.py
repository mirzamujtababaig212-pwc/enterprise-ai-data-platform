from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field
from typing import Any


class MCPServerResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    server_id: str
    transport: str
    connected: bool


class MCPServerListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    servers: list[MCPServerResponse]


class MCPServerHealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    server_id: str
    status: str
    latency_ms: float | None
    last_check: datetime
    error: str | None


class MCPServerHealthListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    servers: list[MCPServerHealthResponse]


class MCPToolResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    input_schema: dict
    metadata: dict
    enabled: bool


class MCPSyncResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    server_id: str
    tools: list[MCPToolResponse]


class MCPServerConfigRequest(BaseModel):
    """Safe MCP configuration accepted by the control-plane API.

    Raw environment values and HTTP headers are intentionally excluded.
    Secrets must be represented through secret references.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    transport: str = Field(min_length=1)

    command: str | None = None
    args: list[str] = Field(default_factory=list)
    cwd: str | None = None

    url: str | None = None

    timeout: float = Field(default=30.0, gt=0)
    read_timeout: float = Field(default=300.0, gt=0)
    health_check_timeout: float = Field(default=5.0, gt=0)
    verify_ssl: bool = True

    recovery_policy: dict[str, Any] = Field(default_factory=dict)
    tool_capabilities: dict[str, dict[str, Any]] = Field(default_factory=dict)


class MCPServerCreateRequest(MCPServerConfigRequest):
    desired_state: str = "active"
    secret_references: dict[str, str] = Field(default_factory=dict)


class MCPServerUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    command: str | None = None
    args: list[str] | None = None
    cwd: str | None = None

    url: str | None = None

    timeout: float | None = Field(default=None, gt=0)
    read_timeout: float | None = Field(default=None, gt=0)
    health_check_timeout: float | None = Field(default=None, gt=0)
    verify_ssl: bool | None = None

    recovery_policy: dict[str, Any] | None = None
    tool_capabilities: dict[str, dict[str, Any]] | None = None

    desired_state: str | None = None
    secret_references: dict[str, str] | None = None


class MCPServerLifecycleResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    server_id: str
    tenant_id: str
    name: str
    transport: str
    desired_state: str

    command: str | None = None
    args: list[str] = Field(default_factory=list)
    cwd: str | None = None
    url: str | None = None

    timeout: float
    read_timeout: float
    health_check_timeout: float
    verify_ssl: bool

    recovery_policy: dict[str, Any]
    tool_capabilities: dict[str, dict[str, Any]]

    secret_references: dict[str, str] = Field(default_factory=dict)

    created_at: datetime
    updated_at: datetime


class MCPServerLifecycleListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    servers: list[MCPServerLifecycleResponse]
