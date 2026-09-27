from datetime import UTC, datetime

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


def test_list_filters_status_and_applies_limit() -> None:
    repository = InMemoryApprovalRequestRepository()

    older = repository.create(
        make_approval("approval-old", "run-1").model_copy(
            update={
                "created_at": datetime(2026, 9, 26, 10, 0, tzinfo=UTC),
            }
        )
    )
    newer = repository.create(
        make_approval("approval-new", "run-1").model_copy(
            update={
                "created_at": datetime(2026, 9, 26, 11, 0, tzinfo=UTC),
            }
        )
    )
    repository.update_status(
        older.approval_id,
        ApprovalStatus.APPROVED,
        resolved_by="reviewer-1",
    )

    pending = repository.list(
        status=ApprovalStatus.PENDING,
        limit=10,
    )
    assert [item.approval_id for item in pending] == [
        "approval-new",
    ]

    scoped = repository.list(limit=1)
    assert [item.approval_id for item in scoped] == [
        newer.approval_id,
    ]


def test_list_rejects_non_positive_limit() -> None:
    repository = InMemoryApprovalRequestRepository()

    with pytest.raises(ValueError, match="limit must be greater than zero"):
        repository.list(limit=0)


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


def make_override(
    override_id: str = "override-1",
    approval_id: str = "approval-1",
    run_id: str = "run-1",
):
    from app.control_plane.approvals.models import ApprovalOverride

    return ApprovalOverride(
        override_id=override_id,
        approval_id=approval_id,
        run_id=run_id,
        actor="operator-1",
        reason="Emergency operational bypass",
    )


def test_override_create_and_get() -> None:
    from app.control_plane.approvals.in_memory import (
        InMemoryApprovalOverrideRepository,
    )

    repository = InMemoryApprovalOverrideRepository()
    override = make_override()

    created = repository.create(override)

    assert created == override
    assert repository.get(override.override_id) == override
    assert repository.get_by_approval(override.approval_id) == override


def test_override_create_rejects_duplicate_override_id() -> None:
    from app.control_plane.approvals.in_memory import (
        InMemoryApprovalOverrideRepository,
    )

    repository = InMemoryApprovalOverrideRepository()
    override = make_override()

    repository.create(override)

    with pytest.raises(
        ValueError,
        match="approval override already exists",
    ):
        repository.create(override)


def test_override_create_rejects_second_override_for_same_approval() -> None:
    from app.control_plane.approvals.in_memory import (
        InMemoryApprovalOverrideRepository,
    )

    repository = InMemoryApprovalOverrideRepository()
    repository.create(make_override())

    with pytest.raises(
        ValueError,
        match="approval override already exists for approval",
    ):
        repository.create(
            make_override(
                override_id="override-2",
                approval_id="approval-1",
            )
        )


def test_override_get_by_approval_returns_none_when_missing() -> None:
    from app.control_plane.approvals.in_memory import (
        InMemoryApprovalOverrideRepository,
    )

    repository = InMemoryApprovalOverrideRepository()

    assert repository.get_by_approval("missing") is None


def test_override_list_by_run() -> None:
    from app.control_plane.approvals.in_memory import (
        InMemoryApprovalOverrideRepository,
    )

    repository = InMemoryApprovalOverrideRepository()

    first = repository.create(
        make_override(
            override_id="override-1",
            approval_id="approval-1",
            run_id="run-1",
        )
    )
    second = repository.create(
        make_override(
            override_id="override-2",
            approval_id="approval-2",
            run_id="run-1",
        )
    )
    repository.create(
        make_override(
            override_id="override-3",
            approval_id="approval-3",
            run_id="run-2",
        )
    )

    assert repository.list_by_run("run-1") == [first, second]


@pytest.mark.parametrize(
    ("method", "value", "message"),
    [
        ("get", " ", "override_id must not be empty"),
        ("get_by_approval", " ", "approval_id must not be empty"),
        ("list_by_run", " ", "run_id must not be empty"),
    ],
)
def test_override_repository_rejects_empty_lookup_values(
    method: str,
    value: str,
    message: str,
) -> None:
    from app.control_plane.approvals.in_memory import (
        InMemoryApprovalOverrideRepository,
    )

    repository = InMemoryApprovalOverrideRepository()

    with pytest.raises(ValueError, match=message):
        getattr(repository, method)(value)
