from __future__ import annotations

from ai_platform.agents.observability import (
    AgentExecutionEvent,
    AgentExecutionEventType,
)
from ai_platform.agents.observer import AgentExecutionObserver
from tools.governance.audit import (
    ToolGovernanceDecisionRecord,
    ToolGovernanceDecisionSink,
)


class ToolGovernanceDecisionObserver(ToolGovernanceDecisionSink):
    """Translate tool governance decisions into agent execution events."""

    def __init__(
        self,
        observer: AgentExecutionObserver,
    ) -> None:
        self._observer = observer

    async def record(
        self,
        record: ToolGovernanceDecisionRecord,
    ) -> None:
        metadata: dict[str, object] = {
            "governance_domain": "tool",
            "decision": record.decision,
        }

        if record.tenant_id is not None:
            metadata["tenant_id"] = record.tenant_id

        if record.reason is not None:
            metadata["reason"] = record.reason

        if record.policy_id is not None:
            metadata["policy_id"] = record.policy_id

        if record.policy_version is not None:
            metadata["policy_version"] = record.policy_version

        if record.details:
            metadata["details"] = dict(record.details)

        await self._observer.record(
            AgentExecutionEvent(
                event_type=AgentExecutionEventType.GOVERNANCE_DECISION,
                agent_name=record.agent_name or "tool.execution",
                run_id=record.run_id,
                session_id=record.session_id,
                user_id=record.principal,
                tool_name=record.tool_name,
                call_id=record.call_id,
                metadata=metadata,
            )
        )
