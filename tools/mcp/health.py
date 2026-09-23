from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum


class MCPHealthState(StrEnum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass(frozen=True)
class MCPHealthStatus:
    """
    Point-in-time health observation for a registered MCP server.
    """

    server_name: str
    status: MCPHealthState
    latency_ms: float
    last_check: datetime
    error: str | None = None
