from __future__ import annotations

from ai_platform.agents.observability import (
    AgentExecutionEvent,
    AgentExecutionEventType,
)
from ai_platform.agents.observer import AgentExecutionObserver
from tools.authorization.audit import (
    ToolAuthorizationAuditRecord,
    ToolAuthorizationAuditSink,
)


class ToolAuthorizationAuditObserver(ToolAuthorizationAuditSink):
    """
    Translate tool authorization decisions into agent execution events.

    The application layer owns the translation so the provider-neutral
    tools package does not depend on the agent observability package.
    """

    def __init__(
        self,
        observer: AgentExecutionObserver,
    ) -> None:
        self._observer = observer

    async def record(
        self,
        record: ToolAuthorizationAuditRecord,
    ) -> None:
        if record.agent_name is None:
            return

        event = AgentExecutionEvent(
            event_type=AgentExecutionEventType.TOOL_AUTHORIZATION_DECISION,
            agent_name=record.agent_name,
            run_id=record.run_id,
            session_id=record.session_id,
            user_id=record.principal,
            tool_name=record.tool_name,
            call_id=record.call_id,
            metadata={
                "allowed": record.allowed,
                "reason": record.reason,
            },
        )

        await self._observer.record(event)
