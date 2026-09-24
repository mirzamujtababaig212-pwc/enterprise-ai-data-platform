from __future__ import annotations

import pytest

from ai_platform.agents.observability import (
    AgentExecutionEvent,
    AgentExecutionEventType,
)
from app.control_plane.agent_run_events.tool_authorization_observer import (
    ToolAuthorizationAuditObserver,
)
from app.control_plane.agent_run_events.tool_governance_observer import (
    ToolGovernanceDecisionObserver,
)
from tools.authorization.audit import ToolAuthorizationAuditRecord
from tools.governance.audit import ToolGovernanceDecisionRecord


class RecordingAgentExecutionObserver:
    def __init__(self) -> None:
        self.events: list[AgentExecutionEvent] = []

    async def record(
        self,
        event: AgentExecutionEvent,
    ) -> None:
        self.events.append(event)


@pytest.mark.asyncio
async def test_authorization_record_is_translated_to_agent_event() -> None:
    observer = RecordingAgentExecutionObserver()
    audit_observer = ToolAuthorizationAuditObserver(observer)

    record = ToolAuthorizationAuditRecord(
        principal="user-123",
        tool_name="rag.search",
        allowed=True,
        reason="Tool is authorized.",
        policy_id="metadata_policy",
        policy_version="1.0",
        run_id="run-123",
        call_id="call-456",
        agent_name="enterprise-rag-analyst",
        session_id="session-789",
    )

    await audit_observer.record(record)

    assert observer.events == [
        AgentExecutionEvent(
            event_type=AgentExecutionEventType.TOOL_AUTHORIZATION_DECISION,
            agent_name="enterprise-rag-analyst",
            run_id="run-123",
            session_id="session-789",
            user_id="user-123",
            tool_name="rag.search",
            call_id="call-456",
            metadata={
                "allowed": True,
                "reason": "Tool is authorized.",
                "policy_id": "metadata_policy",
                "policy_version": "1.0",
            },
        )
    ]


@pytest.mark.asyncio
async def test_authorization_record_persists_principal_as_user_id() -> None:
    observer = RecordingAgentExecutionObserver()
    audit_observer = ToolAuthorizationAuditObserver(observer)

    record = ToolAuthorizationAuditRecord(
        principal="sensitive-user-id",
        tool_name="rag.search",
        allowed=False,
        reason="Tool is not authorized for this principal.",
        run_id="run-123",
        call_id="call-456",
        agent_name="enterprise-rag-analyst",
    )

    await audit_observer.record(record)

    event = observer.events[0]

    assert event.user_id == "sensitive-user-id"
    assert "principal" not in event.metadata
    assert "sensitive-user-id" not in event.metadata
    assert event.event_type is AgentExecutionEventType.TOOL_AUTHORIZATION_DECISION
    assert event.metadata == {
        "allowed": False,
        "reason": "Tool is not authorized for this principal.",
    }


@pytest.mark.asyncio
async def test_authorization_record_without_agent_is_skipped() -> None:
    observer = RecordingAgentExecutionObserver()
    audit_observer = ToolAuthorizationAuditObserver(observer)

    record = ToolAuthorizationAuditRecord(
        principal="user-123",
        tool_name="rag.search",
        allowed=True,
        run_id="run-123",
    )

    await audit_observer.record(record)

    assert observer.events == []


@pytest.mark.asyncio
async def test_tool_governance_record_is_translated_to_agent_event() -> None:
    observer = RecordingAgentExecutionObserver()
    governance_observer = ToolGovernanceDecisionObserver(observer)

    record = ToolGovernanceDecisionRecord(
        tool_name="rag.search",
        decision="deny",
        tenant_id="tenant-123",
        reason="Tool is blocked by tenant policy.",
        policy_id="tenant_policy",
        policy_version="2.1",
        run_id="run-123",
        call_id="call-456",
        agent_name="enterprise-rag-analyst",
        session_id="session-789",
        principal="user-123",
        details={
            "enforcement_layer": "tenant_policy",
            "failure_category": "tenant_policy",
        },
    )

    await governance_observer.record(record)

    assert observer.events == [
        AgentExecutionEvent(
            event_type=AgentExecutionEventType.GOVERNANCE_DECISION,
            agent_name="enterprise-rag-analyst",
            run_id="run-123",
            session_id="session-789",
            user_id="user-123",
            tool_name="rag.search",
            call_id="call-456",
            metadata={
                "governance_domain": "tool",
                "decision": "deny",
                "tenant_id": "tenant-123",
                "reason": "Tool is blocked by tenant policy.",
                "policy_id": "tenant_policy",
                "policy_version": "2.1",
                "details": {
                    "enforcement_layer": "tenant_policy",
                    "failure_category": "tenant_policy",
                },
            },
        )
    ]


@pytest.mark.asyncio
async def test_tool_governance_record_does_not_copy_principal_into_metadata() -> None:
    observer = RecordingAgentExecutionObserver()
    governance_observer = ToolGovernanceDecisionObserver(observer)

    record = ToolGovernanceDecisionRecord(
        tool_name="vehicle.query",
        decision="allow",
        tenant_id="tenant-123",
        run_id="run-123",
        principal="sensitive-user-id",
        details={"enforcement_layer": "authorization"},
    )

    await governance_observer.record(record)

    event = observer.events[0]

    assert event.event_type is AgentExecutionEventType.GOVERNANCE_DECISION
    assert event.user_id == "sensitive-user-id"
    assert "principal" not in event.metadata
    assert "sensitive-user-id" not in event.metadata
