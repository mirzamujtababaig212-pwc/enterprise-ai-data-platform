from __future__ import annotations

import pytest

from app.control_plane.approvals.coordinator import (
    ControlPlaneApprovalCoordinator,
)
from app.control_plane.approvals.in_memory import (
    InMemoryApprovalRequestRepository,
)
from app.control_plane.approvals.policy import SideEffectApprovalPolicy
from app.control_plane.approvals.models import ApprovalStatus
from tools.execution.approval import (
    ToolApprovalDisposition,
    ToolApprovalRequest,
)


def make_request(
    *,
    risk_tier: str = "high",
    side_effect: bool = True,
) -> ToolApprovalRequest:
    return ToolApprovalRequest(
        tool_name="send_notification",
        tool_metadata={
            "risk_tier": risk_tier,
            "side_effect": side_effect,
            "capability": "external_notification",
            "permission_scope": "notifications:send",
        },
        arguments={"recipient": "user@example.com"},
        run_id="run-001",
        step_id="step-001",
        call_id="call-001",
        agent_name="approval-agent",
        session_id="session-001",
        user_id="user-001",
        principal="principal-001",
        tenant_id="tenant-001",
    )


def make_coordinator(
    repository: InMemoryApprovalRequestRepository,
) -> ControlPlaneApprovalCoordinator:
    return ControlPlaneApprovalCoordinator(
        policy=SideEffectApprovalPolicy(
            approval_risk_tiers={"high", "critical"},
        ),
        repository_factory=lambda: repository,
    )


@pytest.mark.asyncio
async def test_not_required_does_not_create_approval() -> None:
    repository = InMemoryApprovalRequestRepository()
    coordinator = make_coordinator(repository)

    decision = await coordinator.evaluate(
        make_request(risk_tier="low"),
    )

    assert decision.disposition is ToolApprovalDisposition.NOT_REQUIRED
    assert decision.approval_id is None
    assert repository.list_by_run("run-001") == []


@pytest.mark.asyncio
async def test_required_creates_pending_approval() -> None:
    repository = InMemoryApprovalRequestRepository()
    coordinator = make_coordinator(repository)

    decision = await coordinator.evaluate(
        make_request(),
    )

    assert decision.disposition is ToolApprovalDisposition.PENDING
    assert decision.approval_id is not None

    approval = repository.get(decision.approval_id)
    assert approval is not None
    assert approval.status is ApprovalStatus.PENDING
    assert approval.run_id == "run-001"
    assert approval.step_id == "step-001"
    assert approval.call_id == "call-001"
    assert approval.tool_name == "send_notification"


@pytest.mark.asyncio
async def test_existing_pending_approval_remains_pending() -> None:
    repository = InMemoryApprovalRequestRepository()
    coordinator = make_coordinator(repository)

    first = await coordinator.evaluate(make_request())
    second = await coordinator.evaluate(make_request())

    assert first.approval_id == second.approval_id
    assert second.disposition is ToolApprovalDisposition.PENDING
    assert len(repository.list_by_run("run-001")) == 1


@pytest.mark.asyncio
async def test_existing_approved_approval_allows_execution() -> None:
    repository = InMemoryApprovalRequestRepository()
    coordinator = make_coordinator(repository)

    first = await coordinator.evaluate(make_request())
    assert first.approval_id is not None

    repository.update_status(
        first.approval_id,
        ApprovalStatus.APPROVED,
        resolved_by="operator-001",
    )

    second = await coordinator.evaluate(make_request())

    assert second.approval_id == first.approval_id
    assert second.disposition is ToolApprovalDisposition.APPROVED


@pytest.mark.asyncio
async def test_existing_rejected_approval_blocks_execution() -> None:
    repository = InMemoryApprovalRequestRepository()
    coordinator = make_coordinator(repository)

    first = await coordinator.evaluate(make_request())
    assert first.approval_id is not None

    repository.update_status(
        first.approval_id,
        ApprovalStatus.REJECTED,
        resolved_by="operator-001",
        resolution_reason="Not authorized for this operation.",
    )

    second = await coordinator.evaluate(make_request())

    assert second.approval_id == first.approval_id
    assert second.disposition is ToolApprovalDisposition.REJECTED


@pytest.mark.asyncio
async def test_approval_id_is_deterministic() -> None:
    repository = InMemoryApprovalRequestRepository()
    coordinator = make_coordinator(repository)

    first = await coordinator.evaluate(make_request(risk_tier="low"))
    second = await coordinator.evaluate(make_request(risk_tier="low"))

    assert first.disposition is ToolApprovalDisposition.NOT_REQUIRED
    assert second.disposition is ToolApprovalDisposition.NOT_REQUIRED

    required_request = make_request()
    third = await coordinator.evaluate(required_request)

    assert third.approval_id is not None

    repository_2 = InMemoryApprovalRequestRepository()
    coordinator_2 = make_coordinator(repository_2)

    fourth = await coordinator_2.evaluate(required_request)

    assert fourth.approval_id == third.approval_id
