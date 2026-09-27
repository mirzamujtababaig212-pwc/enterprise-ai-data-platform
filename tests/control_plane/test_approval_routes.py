from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from ai_platform.agents.models import AgentResponse

from app.control_plane.agent_runs.exceptions import AgentRunAccessDeniedError
from app.control_plane.approvals.models import ApprovalRequest, ApprovalStatus
from app.control_plane.dependencies import (
    get_agent_run_application_service,
    get_agent_run_approval_continuation_service,
    get_approval_request_repository,
)
from app.control_plane.routes.approvals import router


class FakeApprovalRepository:
    def __init__(self) -> None:
        self.approval = ApprovalRequest(
            approval_id="approval-123",
            run_id="run-123",
            step_id="step-1",
            call_id="call-1",
            tool_name="vehicle.lookup",
            idempotency_key="idem-1",
            status=ApprovalStatus.PENDING,
            policy_name="high-risk-tool",
            policy_version="1",
            risk_tier="high",
            requested_action="lookup vehicle",
        )

    def get(self, approval_id: str) -> ApprovalRequest | None:
        if approval_id == self.approval.approval_id:
            return self.approval
        return None


class FakeAgentRunApplicationService:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str | None, str | None]] = []
        self.error: Exception | None = None

    def get_run(
        self,
        run_id: str,
        *,
        tenant_id: str | None = None,
        principal: str | None = None,
    ):
        self.calls.append((run_id, tenant_id, principal))

        if self.error is not None:
            raise self.error

        if tenant_id != "tenant-acme":
            return None

        if principal != "api_key:test-owner":
            raise AgentRunAccessDeniedError(
                f"Principal is not authorized to access agent run '{run_id}'.",
            )

        return object()


class FakeApprovalContinuationService:
    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.error: Exception | None = None

    async def continue_approval(
        self,
        approval_id: str,
        *,
        status: ApprovalStatus,
        resolved_by: str | None = None,
        resolution_reason: str | None = None,
    ) -> AgentResponse:
        self.calls.append(
            (
                approval_id,
                status,
                resolved_by,
                resolution_reason,
            ),
        )

        if self.error is not None:
            raise self.error

        return AgentResponse(
            agent_name="enterprise-analyst",
            output="Approval decision resumed the agent.",
            session_id="session-123",
            metadata={
                "approval_id": approval_id,
                "decision": status.value,
            },
        )

    async def override_approval(
        self,
        approval_id: str,
        *,
        actor: str,
        reason: str,
    ) -> AgentResponse:
        self.calls.append(
            (
                approval_id,
                actor,
                reason,
            ),
        )

        if self.error is not None:
            raise self.error

        return AgentResponse(
            agent_name="enterprise-analyst",
            output="Approval override resumed the agent.",
            session_id="session-123",
            metadata={
                "approval_id": approval_id,
                "decision": "override",
            },
        )


def build_client(
    approval_repository: FakeApprovalRepository,
    application_service: FakeAgentRunApplicationService,
    continuation_service: FakeApprovalContinuationService,
    *,
    principal: str | None = "api_key:test-owner",
    tenant_id: str | None = "tenant-acme",
) -> TestClient:
    app = FastAPI()
    app.include_router(router)

    app.dependency_overrides[get_approval_request_repository] = lambda: approval_repository
    app.dependency_overrides[get_agent_run_application_service] = lambda: application_service
    app.dependency_overrides[get_agent_run_approval_continuation_service] = (
        lambda: continuation_service
    )

    class IdentityMiddleware:
        def __init__(self, inner_app):
            self.inner_app = inner_app

        async def __call__(self, scope, receive, send):
            state = scope.setdefault("state", {})

            if principal is not None:
                state["principal"] = principal

            if tenant_id is not None:
                state["tenant_id"] = tenant_id

            await self.inner_app(scope, receive, send)

    app.add_middleware(IdentityMiddleware)

    return TestClient(app)


def test_approve_approval_uses_authenticated_principal() -> None:
    approval_repository = FakeApprovalRepository()
    application_service = FakeAgentRunApplicationService()
    continuation_service = FakeApprovalContinuationService()

    client = build_client(
        approval_repository,
        application_service,
        continuation_service,
    )

    response = client.post(
        "/api/v1/approvals/approval-123/approve",
        json={"reason": "Looks safe."},
    )

    assert response.status_code == 200
    assert response.json() == {
        "run_id": "run-123",
        "agent_name": "enterprise-analyst",
        "output": "Approval decision resumed the agent.",
        "session_id": "session-123",
        "metadata": {
            "approval_id": "approval-123",
            "decision": "approved",
        },
    }

    assert application_service.calls == [
        ("run-123", "tenant-acme", "api_key:test-owner"),
    ]
    assert continuation_service.calls == [
        (
            "approval-123",
            ApprovalStatus.APPROVED,
            "api_key:test-owner",
            "Looks safe.",
        ),
    ]


def test_reject_approval_uses_authenticated_principal() -> None:
    approval_repository = FakeApprovalRepository()
    application_service = FakeAgentRunApplicationService()
    continuation_service = FakeApprovalContinuationService()

    client = build_client(
        approval_repository,
        application_service,
        continuation_service,
    )

    response = client.post(
        "/api/v1/approvals/approval-123/reject",
        json={"reason": "Policy requires rejection."},
    )

    assert response.status_code == 200

    assert continuation_service.calls == [
        (
            "approval-123",
            ApprovalStatus.REJECTED,
            "api_key:test-owner",
            "Policy requires rejection.",
        ),
    ]


def test_override_approval_uses_authenticated_principal_as_actor() -> None:
    approval_repository = FakeApprovalRepository()
    application_service = FakeAgentRunApplicationService()
    continuation_service = FakeApprovalContinuationService()

    client = build_client(
        approval_repository,
        application_service,
        continuation_service,
    )

    response = client.post(
        "/api/v1/approvals/approval-123/override",
        json={"reason": "Emergency operator override."},
    )

    assert response.status_code == 200

    assert continuation_service.calls == [
        (
            "approval-123",
            "api_key:test-owner",
            "Emergency operator override.",
        ),
    ]


def test_approval_route_denies_wrong_principal() -> None:
    approval_repository = FakeApprovalRepository()
    application_service = FakeAgentRunApplicationService()
    application_service.error = AgentRunAccessDeniedError(
        "Principal is not authorized to access agent run 'run-123'.",
    )
    continuation_service = FakeApprovalContinuationService()

    client = build_client(
        approval_repository,
        application_service,
        continuation_service,
    )

    response = client.post(
        "/api/v1/approvals/approval-123/approve",
        json={"reason": "Attempted unauthorized approval."},
    )

    assert response.status_code == 403
    assert response.json() == {
        "detail": "Principal is not authorized to access agent run 'run-123'.",
    }
    assert continuation_service.calls == []


def test_approval_route_hides_wrong_tenant_approval() -> None:
    approval_repository = FakeApprovalRepository()
    application_service = FakeAgentRunApplicationService()
    continuation_service = FakeApprovalContinuationService()

    client = build_client(
        approval_repository,
        application_service,
        continuation_service,
        tenant_id="tenant-other",
    )

    response = client.post(
        "/api/v1/approvals/approval-123/approve",
        json={"reason": "Cross-tenant attempt."},
    )

    assert response.status_code == 404
    assert response.json() == {
        "detail": "Approval request 'approval-123' was not found.",
    }
    assert continuation_service.calls == []


def test_approval_route_returns_404_for_missing_approval() -> None:
    approval_repository = FakeApprovalRepository()
    application_service = FakeAgentRunApplicationService()
    continuation_service = FakeApprovalContinuationService()

    client = build_client(
        approval_repository,
        application_service,
        continuation_service,
    )

    response = client.post(
        "/api/v1/approvals/missing-approval/approve",
        json={"reason": "Missing approval."},
    )

    assert response.status_code == 404
    assert response.json() == {
        "detail": "Approval request 'missing-approval' was not found.",
    }
    assert application_service.calls == []
    assert continuation_service.calls == []


def test_override_approval_maps_unauthorized_override_to_403() -> None:
    approval_repository = FakeApprovalRepository()
    application_service = FakeAgentRunApplicationService()
    continuation_service = FakeApprovalContinuationService()
    continuation_service.error = PermissionError(
        "Principal is not authorized to override approvals.",
    )

    client = build_client(
        approval_repository,
        application_service,
        continuation_service,
    )

    response = client.post(
        "/api/v1/approvals/approval-123/override",
        json={"reason": "Override requested."},
    )

    assert response.status_code == 403
    assert response.json() == {
        "detail": "Principal is not authorized to override approvals.",
    }


def test_approval_route_requires_authenticated_identity() -> None:
    approval_repository = FakeApprovalRepository()
    application_service = FakeAgentRunApplicationService()
    continuation_service = FakeApprovalContinuationService()

    client = build_client(
        approval_repository,
        application_service,
        continuation_service,
        principal=None,
        tenant_id=None,
    )

    response = client.post(
        "/api/v1/approvals/approval-123/approve",
        json={"reason": "No identity."},
    )

    assert response.status_code == 403
    assert response.json() == {
        "detail": "Authenticated tenant and principal context are required.",
    }
    assert application_service.calls == []
    assert continuation_service.calls == []


def test_override_approval_requires_reason() -> None:
    approval_repository = FakeApprovalRepository()
    application_service = FakeAgentRunApplicationService()
    continuation_service = FakeApprovalContinuationService()

    client = build_client(
        approval_repository,
        application_service,
        continuation_service,
    )

    response = client.post(
        "/api/v1/approvals/approval-123/override",
        json={},
    )

    assert response.status_code == 422
    assert continuation_service.calls == []
