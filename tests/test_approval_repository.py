import pytest

from app.control_plane.approvals.in_memory import (
    InMemoryApprovalRequestRepository,
)
from app.control_plane.approvals.models import ApprovalRequest, ApprovalStatus


def make_approval(
    approval_id: str = "approval-1",
    run_id: str = "run-1",
) -> ApprovalRequest:
    return ApprovalRequest(
        approval_id=approval_id,
        run_id=run_id,
        step_id="step-1",
        call_id="call-1",
        tool_name="send_payment",
        idempotency_key=f"idem-{approval_id}",
        policy_name="high-risk-tool-approval",
        requested_action="Execute payment",
    )


def test_create_and_get() -> None:
    repository = InMemoryApprovalRequestRepository()
    approval = make_approval()

    created = repository.create(approval)

    assert created == approval
    assert repository.get(approval.approval_id) == approval


def test_create_rejects_duplicate_approval_id() -> None:
    repository = InMemoryApprovalRequestRepository()
    approval = make_approval()

    repository.create(approval)

    with pytest.raises(
        ValueError,
        match="approval request already exists",
    ):
        repository.create(approval)


def test_update_status_uses_domain_transition() -> None:
    repository = InMemoryApprovalRequestRepository()
    approval = repository.create(make_approval())

    resolved = repository.update_status(
        approval.approval_id,
        ApprovalStatus.APPROVED,
        resolved_by="reviewer-1",
        resolution_reason="Approved",
    )

    assert resolved.status is ApprovalStatus.APPROVED
    assert repository.get(approval.approval_id) == resolved


def test_update_status_rejects_missing_approval() -> None:
    repository = InMemoryApprovalRequestRepository()

    with pytest.raises(
        KeyError,
        match="approval request not found",
    ):
        repository.update_status(
            "missing",
            ApprovalStatus.APPROVED,
            resolved_by="reviewer-1",
        )


def test_update_status_cannot_resolve_twice() -> None:
    repository = InMemoryApprovalRequestRepository()
    approval = repository.create(make_approval())

    repository.update_status(
        approval.approval_id,
        ApprovalStatus.REJECTED,
        resolved_by="reviewer-1",
    )

    with pytest.raises(
        ValueError,
        match="can only be resolved from pending status",
    ):
        repository.update_status(
            approval.approval_id,
            ApprovalStatus.APPROVED,
            resolved_by="reviewer-2",
        )


def test_list_by_run() -> None:
    repository = InMemoryApprovalRequestRepository()

    first = repository.create(make_approval("approval-1", "run-1"))
    second = repository.create(make_approval("approval-2", "run-1"))
    repository.create(make_approval("approval-3", "run-2"))

    assert repository.list_by_run("run-1") == [first, second]


def test_get_rejects_empty_id() -> None:
    repository = InMemoryApprovalRequestRepository()

    with pytest.raises(ValueError, match="approval_id must not be empty"):
        repository.get(" ")


def test_list_by_run_rejects_empty_run_id() -> None:
    repository = InMemoryApprovalRequestRepository()

    with pytest.raises(ValueError, match="run_id must not be empty"):
        repository.list_by_run(" ")
