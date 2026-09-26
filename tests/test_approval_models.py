from datetime import datetime, timezone

import pytest

from app.control_plane.approvals.models import ApprovalRequest, ApprovalStatus


def make_approval() -> ApprovalRequest:
    return ApprovalRequest(
        approval_id="approval-1",
        run_id="run-1",
        step_id="step-1",
        call_id="call-1",
        tool_name="send_payment",
        idempotency_key="idem-1",
        policy_name="high-risk-tool-approval",
        policy_version="1.0.0",
        risk_tier="high",
        requested_action="Execute payment",
        policy_metadata={"source": "tenant-policy"},
    )


def test_approval_request_defaults_to_pending() -> None:
    approval = make_approval()

    assert approval.status is ApprovalStatus.PENDING
    assert approval.resolved_by is None
    assert approval.resolved_at is None


def test_approval_request_can_be_approved() -> None:
    approval = make_approval()
    resolved_at = datetime(2026, 9, 26, 16, 30, tzinfo=timezone.utc)

    resolved = approval.resolve(
        ApprovalStatus.APPROVED,
        resolved_by="user-123",
        resolution_reason="Approved for execution",
        resolved_at=resolved_at,
    )

    assert resolved.status is ApprovalStatus.APPROVED
    assert resolved.resolved_by == "user-123"
    assert resolved.resolution_reason == "Approved for execution"
    assert resolved.resolved_at == resolved_at
    assert resolved.updated_at == resolved_at
    assert resolved.approval_id == approval.approval_id
    assert resolved.call_id == approval.call_id
    assert resolved.idempotency_key == approval.idempotency_key


def test_approval_request_can_be_rejected() -> None:
    approval = make_approval()

    resolved = approval.resolve(
        ApprovalStatus.REJECTED,
        resolved_by="user-123",
        resolution_reason="Operation not authorized",
    )

    assert resolved.status is ApprovalStatus.REJECTED
    assert resolved.resolved_by == "user-123"
    assert resolved.resolution_reason == "Operation not authorized"
    assert resolved.resolved_at is not None
    assert resolved.resolved_at.tzinfo is not None


@pytest.mark.parametrize(
    "status",
    [
        ApprovalStatus.APPROVED,
        ApprovalStatus.REJECTED,
    ],
)
def test_approval_request_cannot_be_resolved_twice(
    status: ApprovalStatus,
) -> None:
    approval = make_approval().resolve(
        status,
        resolved_by="user-123",
    )

    with pytest.raises(
        ValueError,
        match="can only be resolved from pending status",
    ):
        approval.resolve(
            ApprovalStatus.APPROVED,
            resolved_by="user-456",
        )


def test_approval_request_rejects_invalid_resolution_status() -> None:
    approval = make_approval()

    with pytest.raises(
        ValueError,
        match="can only be resolved as approved or rejected",
    ):
        approval.resolve(
            ApprovalStatus.PENDING,
            resolved_by="user-123",
        )


def test_approval_request_requires_resolver() -> None:
    approval = make_approval()

    with pytest.raises(
        ValueError,
        match="resolved_by must not be empty",
    ):
        approval.resolve(
            ApprovalStatus.APPROVED,
            resolved_by="   ",
        )
