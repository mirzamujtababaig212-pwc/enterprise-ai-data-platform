from __future__ import annotations

import hashlib

from fastapi import FastAPI
from fastapi.testclient import TestClient

from ai_platform.agents.models import AgentResponse

from app.control_plane.agent_run_steps.models import (
    AgentRunStep,
    AgentRunStepStatus,
)
from app.control_plane.routes.agents import router
from app.control_plane.agent_runs.exceptions import (
    AgentRunAccessDeniedError,
)
from app.control_plane.agent_runs.models import (
    AgentRun,
    AgentRunExecutionResult,
    AgentRunStatus,
)
from app.control_plane.dependencies import (
    get_agent_run_application_service,
    get_agent_run_recovery_service,
)


class FakeAgentRunApplicationService:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.requests = []
        self.error: Exception | None = None
        self.steps: list[AgentRunStep] = []
        self.step: AgentRunStep | None = None
        self.step_calls: list[tuple] = []
        self.run = AgentRun(
            run_id="run-cancel-123",
            agent_name="enterprise-analyst",
            principal="api_key:test-owner",
            status=AgentRunStatus.RUNNING,
        )

    async def execute(
        self,
        *,
        agent_name: str,
        request,
        idempotency_key: str | None = None,
    ) -> AgentRunExecutionResult:
        self.requests.append(request)

        if self.error is not None:
            raise self.error

        return AgentRunExecutionResult(
            run_id="run-execute-123",
            response=AgentResponse(
                agent_name=agent_name,
                output="Executed agent response.",
                session_id=request.session_id,
                metadata={},
            ),
        )

    def list_steps(
        self,
        run_id: str,
        *,
        principal: str | None,
        status: AgentRunStepStatus | None = None,
        limit: int = 100,
    ) -> list[AgentRunStep]:
        self.step_calls.append(
            ("list", run_id, principal, status, limit),
        )

        if self.error is not None:
            raise self.error

        if status is None:
            return self.steps

        return [step for step in self.steps if step.status is status]

    def get_step(
        self,
        run_id: str,
        step_id: str,
        *,
        principal: str | None,
    ) -> AgentRunStep | None:
        self.step_calls.append(
            ("get", run_id, step_id, principal),
        )

        if self.error is not None:
            raise self.error

        if self.step is not None:
            if self.step.run_id == run_id and self.step.step_id == step_id:
                return self.step

        return None

    def cancel(self, run_id: str) -> AgentRun:
        self.calls.append(run_id)

        if self.error is not None:
            raise self.error

        return self.run


class FakeAgentRunRecoveryService:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.error: Exception | None = None

    async def recover(
        self,
        run_id: str,
    ) -> AgentRunExecutionResult:
        self.calls.append(run_id)

        if self.error is not None:
            raise self.error

        return AgentRunExecutionResult(
            run_id=run_id,
            response=AgentResponse(
                agent_name="enterprise-analyst",
                output="Recovered agent response.",
                session_id="session-recovered",
                metadata={
                    "provider": "mock",
                    "model": "mock-gpt",
                    "tool_rounds": 1,
                },
            ),
        )


def build_client(
    service: FakeAgentRunRecoveryService,
    application_service: FakeAgentRunApplicationService | None = None,
    *,
    principal: str | None = None,
    tenant_id: str | None = None,
) -> TestClient:
    app = FastAPI()
    app.include_router(router)

    app.dependency_overrides[get_agent_run_recovery_service] = lambda: service

    if application_service is not None:
        app.dependency_overrides[get_agent_run_application_service] = lambda: application_service

    if principal is not None or tenant_id is not None:

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


def test_run_agent_propagates_authenticated_principal() -> None:
    recovery_service = FakeAgentRunRecoveryService()
    application_service = FakeAgentRunApplicationService()
    expected_digest = hashlib.sha256(
        b"super-secret-key",
    ).hexdigest()

    client = build_client(
        recovery_service,
        application_service,
        principal=f"api_key:{expected_digest}",
    )

    response = client.post(
        "/api/v1/agents/enterprise-analyst/run",
        json={
            "input": "Analyze the vehicle data.",
            "session_id": "session-123",
            "user_id": "user-456",
            "metadata": {
                "classification": "internal",
            },
        },
        headers={
            "x-api-key": "super-secret-key",
        },
    )

    assert response.status_code == 200
    assert len(application_service.requests) == 1

    agent_request = application_service.requests[0]

    assert agent_request.principal == f"api_key:{expected_digest}"
    assert agent_request.principal != "super-secret-key"
    assert agent_request.user_id == "user-456"
    assert agent_request.session_id == "session-123"


def test_run_agent_uses_authenticated_tenant_not_metadata_tenant() -> None:
    recovery_service = FakeAgentRunRecoveryService()
    application_service = FakeAgentRunApplicationService()

    client = build_client(
        recovery_service,
        application_service,
        principal="api_key:authenticated-principal",
        tenant_id="tenant-acme",
    )

    response = client.post(
        "/api/v1/agents/enterprise-analyst/run",
        json={
            "input": "Analyze the vehicle data.",
            "session_id": "session-tenant-123",
            "user_id": "user-tenant-456",
            "metadata": {
                "tenant_id": "tenant-attacker",
                "classification": "internal",
            },
        },
    )

    assert response.status_code == 200
    assert len(application_service.requests) == 1

    agent_request = application_service.requests[0]

    assert agent_request.tenant_id == "tenant-acme"
    assert agent_request.metadata["tenant_id"] == "tenant-attacker"
    assert agent_request.tenant_id != agent_request.metadata["tenant_id"]


def test_recover_agent_run_returns_recovered_response() -> None:
    service = FakeAgentRunRecoveryService()
    client = build_client(service)

    response = client.post(
        "/api/v1/agents/runs/run-recovery-123/recover",
    )

    assert response.status_code == 200

    assert response.json() == {
        "run_id": "run-recovery-123",
        "agent_name": "enterprise-analyst",
        "output": "Recovered agent response.",
        "session_id": "session-recovered",
        "metadata": {
            "provider": "mock",
            "model": "mock-gpt",
            "tool_rounds": 1,
        },
    }

    assert service.calls == ["run-recovery-123"]


def test_recover_agent_run_maps_missing_run_to_404() -> None:
    service = FakeAgentRunRecoveryService()
    service.error = LookupError(
        "Agent run 'missing-run' was not found.",
    )
    client = build_client(service)

    response = client.post(
        "/api/v1/agents/runs/missing-run/recover",
    )

    assert response.status_code == 404
    assert response.json() == {
        "detail": "Agent run 'missing-run' was not found.",
    }

    assert service.calls == ["missing-run"]


def test_recover_agent_run_maps_invalid_status_to_422() -> None:
    service = FakeAgentRunRecoveryService()
    service.error = ValueError(
        "Agent run 'completed-run' is not eligible for recovery " "from status 'completed'.",
    )
    client = build_client(service)

    response = client.post(
        "/api/v1/agents/runs/completed-run/recover",
    )

    assert response.status_code == 422
    assert response.json() == {
        "detail": (
            "Agent run 'completed-run' is not eligible for recovery " "from status 'completed'."
        ),
    }

    assert service.calls == ["completed-run"]


def test_recover_agent_run_maps_recovery_failure_to_409() -> None:
    service = FakeAgentRunRecoveryService()
    service.error = RuntimeError(
        "Agent run 'failed-run' has no execution checkpoint " "and cannot be recovered.",
    )
    client = build_client(service)

    response = client.post(
        "/api/v1/agents/runs/failed-run/recover",
    )

    assert response.status_code == 409
    assert response.json() == {
        "detail": (
            "Agent run 'failed-run' has no execution checkpoint " "and cannot be recovered."
        ),
    }

    assert service.calls == ["failed-run"]


def test_cancel_agent_run_returns_202() -> None:
    recovery_service = FakeAgentRunRecoveryService()
    application_service = FakeAgentRunApplicationService()
    client = build_client(
        recovery_service,
        application_service,
    )

    response = client.post(
        "/api/v1/agents/runs/run-cancel-123/cancel",
    )

    assert response.status_code == 202
    assert response.json() == {
        "run_id": "run-cancel-123",
        "status": "running",
        "message": "Cancellation requested.",
    }

    assert application_service.calls == ["run-cancel-123"]


def test_cancel_agent_run_maps_missing_run_to_404() -> None:
    recovery_service = FakeAgentRunRecoveryService()
    application_service = FakeAgentRunApplicationService()
    application_service.error = LookupError(
        "Agent run 'missing-run' was not found.",
    )
    client = build_client(
        recovery_service,
        application_service,
    )

    response = client.post(
        "/api/v1/agents/runs/missing-run/cancel",
    )

    assert response.status_code == 404
    assert response.json() == {
        "detail": "Agent run 'missing-run' was not found.",
    }


def test_cancel_agent_run_maps_non_running_run_to_409() -> None:
    recovery_service = FakeAgentRunRecoveryService()
    application_service = FakeAgentRunApplicationService()
    application_service.error = ValueError(
        "Agent run 'completed-run' is not cancellable from status 'completed'.",
    )
    client = build_client(
        recovery_service,
        application_service,
    )

    response = client.post(
        "/api/v1/agents/runs/completed-run/cancel",
    )

    assert response.status_code == 409
    assert response.json() == {
        "detail": ("Agent run 'completed-run' is not cancellable " "from status 'completed'."),
    }


def test_cancel_agent_run_maps_missing_active_task_to_409() -> None:
    recovery_service = FakeAgentRunRecoveryService()
    application_service = FakeAgentRunApplicationService()
    application_service.error = RuntimeError(
        "Agent run 'running-run' is running but has no active " "execution task in this process.",
    )
    client = build_client(
        recovery_service,
        application_service,
    )

    response = client.post(
        "/api/v1/agents/runs/running-run/cancel",
    )

    assert response.status_code == 409
    assert response.json() == {
        "detail": (
            "Agent run 'running-run' is running but has no active "
            "execution task in this process."
        ),
    }


def _route_step(
    *,
    run_id: str = "run-steps-123",
    step_id: str = "step-1",
    step_index: int = 0,
    status: AgentRunStepStatus = AgentRunStepStatus.COMPLETED,
) -> AgentRunStep:
    return AgentRunStep(
        run_id=run_id,
        step_id=step_id,
        step_index=step_index,
        step_type="tool_execution",
        status=status,
        attempt=2,
        tool_name="vehicle_query",
        call_id="call-1",
        input={"query": "vehicle events"},
        output={"rows": 3},
        error=None,
        failure_category=None,
        started_at=None,
        completed_at=None,
        metadata={
            "source": "test",
            "classification": "internal",
        },
    )


def test_list_agent_run_steps_returns_steps() -> None:
    recovery_service = FakeAgentRunRecoveryService()
    application_service = FakeAgentRunApplicationService()
    application_service.steps = [
        _route_step(
            step_id="step-1",
            step_index=0,
            status=AgentRunStepStatus.COMPLETED,
        ),
        _route_step(
            step_id="step-2",
            step_index=1,
            status=AgentRunStepStatus.FAILED,
        ),
    ]

    client = build_client(
        recovery_service,
        application_service,
        principal="api_key:test-owner",
    )

    response = client.get(
        "/api/v1/agents/runs/run-steps-123/steps",
    )

    assert response.status_code == 200
    assert response.json() == {
        "steps": [
            {
                "run_id": "run-steps-123",
                "step_id": "step-1",
                "step_index": 0,
                "step_type": "tool_execution",
                "status": "completed",
                "attempt": 2,
                "tool_name": "vehicle_query",
                "call_id": "call-1",
                "input": {"query": "vehicle events"},
                "output": {"rows": 3},
                "error": None,
                "failure_category": None,
                "started_at": None,
                "completed_at": None,
                "created_at": None,
                "updated_at": None,
                "metadata": {
                    "source": "test",
                    "classification": "internal",
                },
            },
            {
                "run_id": "run-steps-123",
                "step_id": "step-2",
                "step_index": 1,
                "step_type": "tool_execution",
                "status": "failed",
                "attempt": 2,
                "tool_name": "vehicle_query",
                "call_id": "call-1",
                "input": {"query": "vehicle events"},
                "output": {"rows": 3},
                "error": None,
                "failure_category": None,
                "started_at": None,
                "completed_at": None,
                "created_at": None,
                "updated_at": None,
                "metadata": {
                    "source": "test",
                    "classification": "internal",
                },
            },
        ],
    }

    assert application_service.step_calls == [
        ("list", "run-steps-123", "api_key:test-owner", None, 100),
    ]


def test_list_agent_run_steps_filters_by_status() -> None:
    recovery_service = FakeAgentRunRecoveryService()
    application_service = FakeAgentRunApplicationService()
    application_service.steps = [
        _route_step(
            step_id="step-completed",
            status=AgentRunStepStatus.COMPLETED,
        ),
        _route_step(
            step_id="step-running",
            status=AgentRunStepStatus.RUNNING,
        ),
    ]

    client = build_client(
        recovery_service,
        application_service,
        principal="api_key:test-owner",
    )

    response = client.get(
        "/api/v1/agents/runs/run-steps-123/steps",
        params={
            "status": "running",
            "limit": 25,
        },
    )

    assert response.status_code == 200
    assert [step["step_id"] for step in response.json()["steps"]] == ["step-running"]

    assert application_service.step_calls == [
        (
            "list",
            "run-steps-123",
            "api_key:test-owner",
            AgentRunStepStatus.RUNNING,
            25,
        ),
    ]


def test_list_agent_run_steps_rejects_invalid_status() -> None:
    recovery_service = FakeAgentRunRecoveryService()
    application_service = FakeAgentRunApplicationService()

    client = build_client(
        recovery_service,
        application_service,
    )

    response = client.get(
        "/api/v1/agents/runs/run-steps-123/steps",
        params={"status": "not-a-status"},
    )

    assert response.status_code == 422
    assert response.json() == {
        "detail": ("Invalid agent run step status: not-a-status"),
    }

    assert application_service.step_calls == []


def test_list_agent_run_steps_maps_missing_run_to_404() -> None:
    recovery_service = FakeAgentRunRecoveryService()
    application_service = FakeAgentRunApplicationService()
    application_service.error = LookupError(
        "Agent run 'missing-run' was not found.",
    )

    client = build_client(
        recovery_service,
        application_service,
        principal="api_key:test-owner",
    )

    response = client.get(
        "/api/v1/agents/runs/missing-run/steps",
    )

    assert response.status_code == 404
    assert response.json() == {
        "detail": "Agent run 'missing-run' was not found.",
    }


def test_get_agent_run_step_returns_step() -> None:
    recovery_service = FakeAgentRunRecoveryService()
    application_service = FakeAgentRunApplicationService()
    application_service.step = _route_step(
        run_id="run-steps-123",
        step_id="step-42",
    )

    client = build_client(
        recovery_service,
        application_service,
        principal="api_key:test-owner",
    )

    response = client.get(
        "/api/v1/agents/runs/run-steps-123/steps/step-42",
    )

    assert response.status_code == 200
    assert response.json() == {
        "run_id": "run-steps-123",
        "step_id": "step-42",
        "step_index": 0,
        "step_type": "tool_execution",
        "status": "completed",
        "attempt": 2,
        "tool_name": "vehicle_query",
        "call_id": "call-1",
        "input": {"query": "vehicle events"},
        "output": {"rows": 3},
        "error": None,
        "failure_category": None,
        "started_at": None,
        "completed_at": None,
        "created_at": None,
        "updated_at": None,
        "metadata": {
            "source": "test",
            "classification": "internal",
        },
    }

    assert application_service.step_calls == [
        ("get", "run-steps-123", "step-42", "api_key:test-owner"),
    ]


def test_get_agent_run_step_maps_missing_step_to_404() -> None:
    recovery_service = FakeAgentRunRecoveryService()
    application_service = FakeAgentRunApplicationService()

    client = build_client(
        recovery_service,
        application_service,
        principal="api_key:test-owner",
    )

    response = client.get(
        "/api/v1/agents/runs/run-steps-123/steps/missing-step",
    )

    assert response.status_code == 404
    assert response.json() == {
        "detail": ("Agent run step 'missing-step' for run " "'run-steps-123' was not found."),
    }

    assert application_service.step_calls == [
        ("get", "run-steps-123", "missing-step", "api_key:test-owner"),
    ]


def test_list_agent_run_steps_maps_access_denied_to_403() -> None:
    recovery_service = FakeAgentRunRecoveryService()
    application_service = FakeAgentRunApplicationService()
    application_service.error = AgentRunAccessDeniedError(
        "Principal is not authorized to access agent run 'run-steps-123'.",
    )

    client = build_client(
        recovery_service,
        application_service,
        principal="api_key:other-user",
    )

    response = client.get(
        "/api/v1/agents/runs/run-steps-123/steps",
    )

    assert response.status_code == 403
    assert response.json() == {
        "detail": ("Principal is not authorized to access agent run " "'run-steps-123'."),
    }


def test_get_agent_run_step_maps_access_denied_to_403() -> None:
    recovery_service = FakeAgentRunRecoveryService()
    application_service = FakeAgentRunApplicationService()
    application_service.error = AgentRunAccessDeniedError(
        "Principal is not authorized to access agent run 'run-steps-123'.",
    )

    client = build_client(
        recovery_service,
        application_service,
        principal="api_key:other-user",
    )

    response = client.get(
        "/api/v1/agents/runs/run-steps-123/steps/step-42",
    )

    assert response.status_code == 403
    assert response.json() == {
        "detail": ("Principal is not authorized to access agent run " "'run-steps-123'."),
    }


def test_get_agent_run_step_maps_missing_run_to_404() -> None:
    recovery_service = FakeAgentRunRecoveryService()
    application_service = FakeAgentRunApplicationService()
    application_service.error = LookupError(
        "Agent run 'missing-run' was not found.",
    )

    client = build_client(
        recovery_service,
        application_service,
        principal="api_key:test-owner",
    )

    response = client.get(
        "/api/v1/agents/runs/missing-run/steps/step-1",
    )

    assert response.status_code == 404
    assert response.json() == {
        "detail": "Agent run 'missing-run' was not found.",
    }
