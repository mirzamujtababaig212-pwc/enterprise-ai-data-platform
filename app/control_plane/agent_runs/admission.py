from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ai_platform.agents.models import AgentRequest

from app.control_plane.agent_runs.models import AgentRun


@dataclass(frozen=True)
class AgentRunAdmissionResult:
    """Immutable result of an agent run admission decision."""

    allowed: bool
    reason: str | None = None


class AgentRunAdmissionPolicy(Protocol):
    """Policy determining whether an agent run may start execution."""

    async def evaluate(
        self,
        *,
        agent_name: str,
        request: AgentRequest,
        run: AgentRun,
    ) -> AgentRunAdmissionResult: ...


class AllowAllAgentRunAdmissionPolicy:
    """Default admission policy preserving the existing execution behavior."""

    async def evaluate(
        self,
        *,
        agent_name: str,
        request: AgentRequest,
        run: AgentRun,
    ) -> AgentRunAdmissionResult:
        del agent_name, request, run
        return AgentRunAdmissionResult(allowed=True)
