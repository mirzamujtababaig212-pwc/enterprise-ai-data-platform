from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from ai_platform.agents.models import AgentResponse

from app.control_plane.routes.agents import router
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
        self.error: Exception | None = None
        self.run = AgentRun(
            run_id="run-cancel-123",
            agent_name="enterprise-analyst",
            status=AgentRunStatus.RUNNING,
        )

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
) -> TestClient:
    app = FastAPI()
    app.include_router(router)

    app.dependency_overrides[get_agent_run_recovery_service] = lambda: service

    if application_service is not None:
        app.dependency_overrides[get_agent_run_application_service] = lambda: application_service

    return TestClient(app)


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
