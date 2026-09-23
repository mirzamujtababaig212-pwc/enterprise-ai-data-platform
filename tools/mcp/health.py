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


@dataclass
class MCPHealthHistory:
    """
    In-memory health history and deterministic state transition tracker.

    Ordinary health failures require two consecutive failures to move
    from DEGRADED to UNHEALTHY. Recovery requires two consecutive
    successful observations to move from UNHEALTHY to HEALTHY.

    Hard failures, such as a disconnected server or an unsupported
    health mechanism, transition directly to UNHEALTHY.
    """

    state: MCPHealthState | None = None
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    last_healthy_at: datetime | None = None
    last_unhealthy_at: datetime | None = None
    last_error: str | None = None

    def record_success(self, checked_at: datetime) -> MCPHealthState:
        self.consecutive_successes += 1
        self.consecutive_failures = 0
        self.last_error = None

        if self.state is None:
            self.state = MCPHealthState.HEALTHY
        elif self.state is MCPHealthState.UNHEALTHY:
            self.state = MCPHealthState.DEGRADED
        elif self.state is MCPHealthState.DEGRADED:
            self.state = MCPHealthState.HEALTHY

        if self.state is MCPHealthState.HEALTHY:
            self.last_healthy_at = checked_at

        return self.state

    def record_failure(
        self,
        checked_at: datetime,
        *,
        error: str | None = None,
        hard_failure: bool = False,
    ) -> MCPHealthState:
        self.consecutive_failures += 1
        self.consecutive_successes = 0
        self.last_error = error

        if hard_failure:
            self.state = MCPHealthState.UNHEALTHY
        elif self.state is None:
            self.state = MCPHealthState.DEGRADED
        elif self.state is MCPHealthState.HEALTHY:
            self.state = MCPHealthState.DEGRADED
        elif self.state is MCPHealthState.DEGRADED:
            self.state = MCPHealthState.UNHEALTHY

        if self.state is MCPHealthState.UNHEALTHY:
            self.last_unhealthy_at = checked_at

        return self.state
