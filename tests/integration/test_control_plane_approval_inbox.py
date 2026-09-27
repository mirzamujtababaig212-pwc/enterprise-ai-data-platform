"""Production integration tests for the approval inbox."""

from __future__ import annotations

import os
from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete

from app.control_plane.agent_runs.models import AgentRun
from app.control_plane.agent_runs.postgres_repository import (
    PostgreSQLAgentRunRepository,
)
from app.control_plane.approvals.models import ApprovalRequest, ApprovalStatus
from app.control_plane.approvals.postgres_repository import (
    PostgreSQLApprovalRequestRepository,
)
from app.control_plane.app import app
from app.control_plane.auth import principal_from_api_key
from app.control_plane.persistence.database import SessionLocal
from app.control_plane.persistence.models import (
    AgentRunRecord,
    ApprovalRequestRecord,
)

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_POSTGRES_INTEGRATION") != "1",
    reason="Set RUN_POSTGRES_INTEGRATION=1 to run the PostgreSQL integration test",
)


API_KEY = os.getenv("API_KEY", "change-me")


def test_production_approval_inbox_scopes_to_authenticated_identity() -> None:
    """Exercise the production approval inbox through the real HTTP stack."""

    client = TestClient(app)
    authenticated_principal = principal_from_api_key(API_KEY)

    matching_run_id = "approval-inbox-match-run"
    other_principal_run_id = "approval-inbox-other-run"

    matching_approval_id = "approval-inbox-match"
    other_approval_id = "approval-inbox-other"

    created_run_ids = {
        matching_run_id,
        other_principal_run_id,
    }
    created_approval_ids = {
        matching_approval_id,
        other_approval_id,
    }

    matching_run = AgentRun(
        run_id=matching_run_id,
        agent_name="approval-inbox-integration-agent",
        session_id="approval-inbox-integration-session",
        user_id="approval-inbox-user",
        principal=authenticated_principal,
        tenant_id="tenant-a",
    )

    other_principal_run = AgentRun(
        run_id=other_principal_run_id,
        agent_name="approval-inbox-integration-agent",
        session_id="approval-inbox-other-session",
        user_id="approval-inbox-other-user",
        principal="api_key:approval-inbox-other-principal",
        tenant_id="tenant-a",
    )

    matching_approval = ApprovalRequest(
        approval_id=matching_approval_id,
        run_id=matching_run_id,
        step_id="step-inbox-match",
        call_id="call-inbox-match",
        tool_name="send_notification",
        idempotency_key="idem-inbox-match",
        status=ApprovalStatus.PENDING,
        policy_name="side-effect-requires-approval",
        policy_version="1.2.0",
        risk_tier="high",
        requested_action="Send an external notification",
        policy_metadata={
            "source": "production-integration-test",
            "requires_human_approval": True,
        },
        created_at=datetime(2026, 9, 26, 15, 0, tzinfo=UTC),
        updated_at=datetime(2026, 9, 26, 15, 0, tzinfo=UTC),
    )

    other_approval = ApprovalRequest(
        approval_id=other_approval_id,
        run_id=other_principal_run_id,
        step_id="step-inbox-other",
        call_id="call-inbox-other",
        tool_name="send_notification",
        idempotency_key="idem-inbox-other",
        status=ApprovalStatus.PENDING,
        policy_name="side-effect-requires-approval",
        policy_version="1.2.0",
        risk_tier="high",
        requested_action="Send another external notification",
        policy_metadata={
            "source": "production-integration-test",
        },
        created_at=datetime(2026, 9, 26, 16, 0, tzinfo=UTC),
        updated_at=datetime(2026, 9, 26, 16, 0, tzinfo=UTC),
    )

    try:
        with SessionLocal() as session:
            run_repository = PostgreSQLAgentRunRepository(session)
            approval_repository = PostgreSQLApprovalRequestRepository(session)

            run_repository.create(matching_run)
            run_repository.create(other_principal_run)

            approval_repository.create(matching_approval)
            approval_repository.create(other_approval)

        response = client.get(
            "/api/v1/approvals",
            headers={
                "x-api-key": API_KEY,
            },
            params={
                "status": "pending",
                "limit": 10,
            },
        )

        assert response.status_code == 200, response.text

        payload = response.json()

        assert len(payload) == 1
        assert payload[0] == {
            "approval_id": matching_approval_id,
            "run_id": matching_run_id,
            "step_id": "step-inbox-match",
            "call_id": "call-inbox-match",
            "tool_name": "send_notification",
            "idempotency_key": "idem-inbox-match",
            "status": "pending",
            "policy_name": "side-effect-requires-approval",
            "policy_version": "1.2.0",
            "risk_tier": "high",
            "requested_action": "Send an external notification",
            "policy_metadata": {
                "source": "production-integration-test",
                "requires_human_approval": True,
            },
            "resolved_by": None,
            "resolution_reason": None,
            "created_at": "2026-09-26T15:00:00Z",
            "updated_at": "2026-09-26T15:00:00Z",
            "resolved_at": None,
        }
    finally:
        with SessionLocal() as session:
            session.execute(
                delete(ApprovalRequestRecord).where(
                    ApprovalRequestRecord.approval_id.in_(created_approval_ids),
                )
            )
            session.execute(
                delete(AgentRunRecord).where(
                    AgentRunRecord.run_id.in_(created_run_ids),
                )
            )
            session.commit()
