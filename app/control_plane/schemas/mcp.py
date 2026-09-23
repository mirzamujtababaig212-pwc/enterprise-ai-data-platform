from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict


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
