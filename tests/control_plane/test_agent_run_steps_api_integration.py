from __future__ import annotations

import os

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import delete, create_engine
from sqlalchemy.orm import sessionmaker

from ai_platform.agents.runtime import AgentRuntime
from app.control_plane.agent_run_steps.models import (
    AgentRunStep,
    AgentRunStepStatus,
)
from app.control_plane.agent_run_steps.postgres_repository import (
    PostgreSQLAgentRunStepsRepository,
)
from app.control_plane.agent_runs.application_service import (
    AgentRunApplicationService,
)
from app.control_plane.agent_runs.models import AgentRun, AgentRunStatus
from app.control_plane.agent_runs.postgres_repository import (
    PostgreSQLAgentRunRepository,
)
from app.control_plane.dependencies import get_agent_run_application_service
from app.control_plane.persistence.models import (
    AgentRunRecord,
    AgentRunStepRecord,
)
from app.control_plane.routes.agents import router


@pytest.fixture()
def postgres_sessions():
    if os.getenv("RUN_POSTGRES_INTEGRATION") != "1":
        pytest.skip("Set RUN_POSTGRES_INTEGRATION=1 to run the PostgreSQL integration test")

    host = os.getenv("POSTGRES_TEST_HOST", "localhost")
    port = os.getenv("POSTGRES_TEST_PORT", "5432")
    user = os.getenv("POSTGRES_TEST_USER", "postgres")
    password = os.getenv("POSTGRES_TEST_PASSWORD", "postgres")
    database = os.getenv("POSTGRES_TEST_DB", "vehicle_platform")

    engine = create_engine(
        f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}",
        pool_pre_ping=True,
        future=True,
    )

    session_factory = sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
    )

    try:
        yield engine, session_factory
    finally:
        engine.dispose()


def make_run(*, run_id: str) -> AgentRun:
    return AgentRun(
        run_id=run_id,
        agent_name="api-step-integration-agent",
        session_id=f"session-{run_id}",
        user_id="api-integration-user",
        principal="api-integration-principal",
        status=AgentRunStatus.COMPLETED,
        recovery_attempts=0,
        metadata={"source": "api-integration-test"},
    )


def make_step(
    *,
    run_id: str,
    step_id: str,
    step_index: int,
    status: AgentRunStepStatus,
    tool_name: str | None,
    call_id: str | None,
    input: object,
    output: object | None = None,
    error: str | None = None,
    failure_category: str | None = None,
    metadata: dict | None = None,
) -> AgentRunStep:
    return AgentRunStep(
        run_id=run_id,
        step_id=step_id,
        step_index=step_index,
        step_type="tool",
        status=status,
        attempt=1,
        tool_name=tool_name,
        call_id=call_id,
        input=input,
        output=output,
        error=error,
        failure_category=failure_category,
        metadata={} if metadata is None else metadata,
    )


def clear_run(session_factory, *, run_id: str) -> None:
    session = session_factory()
    try:
        session.execute(
            delete(AgentRunStepRecord).where(
                AgentRunStepRecord.run_id == run_id,
            )
        )
        session.execute(
            delete(AgentRunRecord).where(
                AgentRunRecord.run_id == run_id,
            )
        )
        session.commit()
    finally:
        session.close()


def build_client(
    session_factory,
) -> TestClient:
    session = session_factory()

    service = AgentRunApplicationService(
        runtime=AgentRuntime.__new__(AgentRuntime),
        repository=PostgreSQLAgentRunRepository(session),
        agent_run_steps_repository=PostgreSQLAgentRunStepsRepository(session),
    )

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_agent_run_application_service] = lambda: service

    class PrincipalMiddleware:
        def __init__(self, inner_app):
            self.inner_app = inner_app

        async def __call__(self, scope, receive, send):
            scope.setdefault("state", {})["principal"] = "api-integration-principal"
            await self.inner_app(scope, receive, send)

    app.add_middleware(PrincipalMiddleware)

    return TestClient(app)


@pytest.mark.asyncio
async def test_agent_run_steps_api_lists_persisted_postgres_steps(
    postgres_sessions,
) -> None:
    _, session_factory = postgres_sessions

    run_id = "api-step-list-integration"
    clear_run(session_factory, run_id=run_id)

    session = session_factory()

    try:
        run_repository = PostgreSQLAgentRunRepository(session)
        step_repository = PostgreSQLAgentRunStepsRepository(session)

        run_repository.create(make_run(run_id=run_id))

        step_repository.create(
            make_step(
                run_id=run_id,
                step_id="step-2",
                step_index=2,
                status=AgentRunStepStatus.FAILED,
                tool_name="vehicle_query",
                call_id="call-2",
                input={"vehicle_id": "V-200"},
                error="query failed",
                failure_category="execution_error",
                metadata={"trace_id": "trace-2"},
            )
        )
        step_repository.create(
            make_step(
                run_id=run_id,
                step_id="step-1",
                step_index=1,
                status=AgentRunStepStatus.COMPLETED,
                tool_name="vehicle_query",
                call_id="call-1",
                input={"vehicle_id": "V-100"},
                output={"rows": 4},
                metadata={"trace_id": "trace-1"},
            )
        )

        session.close()

        client = build_client(session_factory)

        response = client.get(
            f"/api/v1/agents/runs/{run_id}/steps",
        )

        assert response.status_code == 200

        payload = response.json()

        assert [step["step_id"] for step in payload["steps"]] == [
            "step-1",
            "step-2",
        ]

        assert payload["steps"][0] == {
            "run_id": run_id,
            "step_id": "step-1",
            "step_index": 1,
            "step_type": "tool",
            "status": "completed",
            "attempt": 1,
            "tool_name": "vehicle_query",
            "call_id": "call-1",
            "input": {"vehicle_id": "V-100"},
            "output": {"rows": 4},
            "error": None,
            "failure_category": None,
            "started_at": None,
            "completed_at": None,
            "created_at": payload["steps"][0]["created_at"],
            "updated_at": payload["steps"][0]["updated_at"],
            "metadata": {"trace_id": "trace-1"},
        }

        assert payload["steps"][1]["status"] == "failed"
        assert payload["steps"][1]["error"] == "query failed"
        assert payload["steps"][1]["failure_category"] == "execution_error"
        assert payload["steps"][1]["metadata"] == {"trace_id": "trace-2"}
    finally:
        if session.is_active:
            session.close()
        clear_run(session_factory, run_id=run_id)


@pytest.mark.asyncio
async def test_agent_run_steps_api_filters_persisted_postgres_steps_by_status(
    postgres_sessions,
) -> None:
    _, session_factory = postgres_sessions

    run_id = "api-step-filter-integration"
    clear_run(session_factory, run_id=run_id)

    session = session_factory()

    try:
        run_repository = PostgreSQLAgentRunRepository(session)
        step_repository = PostgreSQLAgentRunStepsRepository(session)

        run_repository.create(make_run(run_id=run_id))

        step_repository.create(
            make_step(
                run_id=run_id,
                step_id="step-completed",
                step_index=1,
                status=AgentRunStepStatus.COMPLETED,
                tool_name="vehicle_query",
                call_id="call-completed",
                input={"vehicle_id": "V-100"},
                output={"rows": 1},
            )
        )
        step_repository.create(
            make_step(
                run_id=run_id,
                step_id="step-failed",
                step_index=2,
                status=AgentRunStepStatus.FAILED,
                tool_name="vehicle_query",
                call_id="call-failed",
                input={"vehicle_id": "V-200"},
                error="query failed",
                failure_category="execution_error",
            )
        )

        session.close()

        client = build_client(session_factory)

        response = client.get(
            f"/api/v1/agents/runs/{run_id}/steps",
            params={"status": "failed"},
        )

        assert response.status_code == 200

        payload = response.json()

        assert [step["step_id"] for step in payload["steps"]] == [
            "step-failed",
        ]
        assert payload["steps"][0]["status"] == "failed"
    finally:
        if session.is_active:
            session.close()
        clear_run(session_factory, run_id=run_id)


@pytest.mark.asyncio
async def test_agent_run_step_api_returns_persisted_postgres_step(
    postgres_sessions,
) -> None:
    _, session_factory = postgres_sessions

    run_id = "api-step-detail-integration"
    clear_run(session_factory, run_id=run_id)

    session = session_factory()

    try:
        run_repository = PostgreSQLAgentRunRepository(session)
        step_repository = PostgreSQLAgentRunStepsRepository(session)

        run_repository.create(make_run(run_id=run_id))

        step_repository.create(
            make_step(
                run_id=run_id,
                step_id="step-detail",
                step_index=3,
                status=AgentRunStepStatus.COMPLETED,
                tool_name="vehicle_query",
                call_id="call-detail",
                input={"vehicle_id": "V-300"},
                output={
                    "rows": 2,
                    "source": "delta",
                },
                metadata={
                    "trace_id": "trace-detail",
                    "release_id": "release-1",
                },
            )
        )

        session.close()

        client = build_client(session_factory)

        response = client.get(
            f"/api/v1/agents/runs/{run_id}/steps/step-detail",
        )

        assert response.status_code == 200

        payload = response.json()

        assert payload["run_id"] == run_id
        assert payload["step_id"] == "step-detail"
        assert payload["step_index"] == 3
        assert payload["step_type"] == "tool"
        assert payload["status"] == "completed"
        assert payload["attempt"] == 1
        assert payload["tool_name"] == "vehicle_query"
        assert payload["call_id"] == "call-detail"
        assert payload["input"] == {"vehicle_id": "V-300"}
        assert payload["output"] == {
            "rows": 2,
            "source": "delta",
        }
        assert payload["metadata"] == {
            "trace_id": "trace-detail",
            "release_id": "release-1",
        }
    finally:
        if session.is_active:
            session.close()
        clear_run(session_factory, run_id=run_id)


@pytest.mark.asyncio
async def test_agent_run_steps_api_rejects_mismatched_principal(
    postgres_sessions,
) -> None:
    _, session_factory = postgres_sessions

    run_id = "api-step-authz-integration"
    clear_run(session_factory, run_id=run_id)

    session = session_factory()

    try:
        run_repository = PostgreSQLAgentRunRepository(session)
        step_repository = PostgreSQLAgentRunStepsRepository(session)

        run_repository.create(make_run(run_id=run_id))

        step_repository.create(
            make_step(
                run_id=run_id,
                step_id="step-authz",
                step_index=0,
                status=AgentRunStepStatus.COMPLETED,
                tool_name="vehicle_query",
                call_id="call-authz",
                input={"vehicle_id": "V-400"},
                output={"rows": 1},
            )
        )

        session.close()

        # Build a dedicated client whose authenticated principal differs
        # from the run owner.
        def build_mismatched_client():
            session = session_factory()

            service = AgentRunApplicationService(
                runtime=AgentRuntime.__new__(AgentRuntime),
                repository=PostgreSQLAgentRunRepository(session),
                agent_run_steps_repository=PostgreSQLAgentRunStepsRepository(
                    session,
                ),
            )

            app = FastAPI()
            app.include_router(router)
            app.dependency_overrides[get_agent_run_application_service] = lambda: service

            class MismatchedPrincipalMiddleware:
                def __init__(self, inner_app):
                    self.inner_app = inner_app

                async def __call__(self, scope, receive, send):
                    scope.setdefault("state", {})["principal"] = "api-integration-other-principal"
                    await self.inner_app(scope, receive, send)

            app.add_middleware(MismatchedPrincipalMiddleware)
            return TestClient(app)

        client = build_mismatched_client()

        response = client.get(
            f"/api/v1/agents/runs/{run_id}/steps",
        )

        assert response.status_code == 403
        assert response.json()["detail"] == (
            f"Principal is not authorized to access agent run '{run_id}'."
        )
    finally:
        if session.is_active:
            session.close()
        clear_run(session_factory, run_id=run_id)


@pytest.mark.asyncio
async def test_agent_run_step_api_rejects_mismatched_principal_for_detail(
    postgres_sessions,
) -> None:
    _, session_factory = postgres_sessions

    run_id = "api-step-detail-authz-integration"
    clear_run(session_factory, run_id=run_id)

    session = session_factory()

    try:
        run_repository = PostgreSQLAgentRunRepository(session)
        step_repository = PostgreSQLAgentRunStepsRepository(session)

        run_repository.create(make_run(run_id=run_id))

        step_repository.create(
            make_step(
                run_id=run_id,
                step_id="step-authz-detail",
                step_index=0,
                status=AgentRunStepStatus.COMPLETED,
                tool_name="vehicle_query",
                call_id="call-authz-detail",
                input={"vehicle_id": "V-500"},
                output={"rows": 1},
            )
        )

        session.close()

        session = session_factory()

        service = AgentRunApplicationService(
            runtime=AgentRuntime.__new__(AgentRuntime),
            repository=PostgreSQLAgentRunRepository(session),
            agent_run_steps_repository=PostgreSQLAgentRunStepsRepository(
                session,
            ),
        )

        app = FastAPI()
        app.include_router(router)
        app.dependency_overrides[get_agent_run_application_service] = lambda: service

        class MismatchedPrincipalMiddleware:
            def __init__(self, inner_app):
                self.inner_app = inner_app

            async def __call__(self, scope, receive, send):
                scope.setdefault("state", {})["principal"] = "api-integration-other-principal"
                await self.inner_app(scope, receive, send)

        app.add_middleware(MismatchedPrincipalMiddleware)

        client = TestClient(app)

        response = client.get(
            f"/api/v1/agents/runs/{run_id}/steps/step-authz-detail",
        )

        assert response.status_code == 403
        assert response.json()["detail"] == (
            f"Principal is not authorized to access agent run '{run_id}'."
        )
    finally:
        if session.is_active:
            session.close()
        clear_run(session_factory, run_id=run_id)


@pytest.mark.asyncio
async def test_agent_run_steps_api_returns_404_for_missing_step(
    postgres_sessions,
) -> None:
    _, session_factory = postgres_sessions

    run_id = "api-step-missing-integration"
    clear_run(session_factory, run_id=run_id)

    session = session_factory()

    try:
        run_repository = PostgreSQLAgentRunRepository(session)
        run_repository.create(make_run(run_id=run_id))
        session.close()

        client = build_client(session_factory)

        response = client.get(
            f"/api/v1/agents/runs/{run_id}/steps/missing-step",
        )

        assert response.status_code == 404
        assert response.json()["detail"] == (
            f"Agent run step 'missing-step' for run '{run_id}' was not found."
        )
    finally:
        if session.is_active:
            session.close()
        clear_run(session_factory, run_id=run_id)


@pytest.mark.asyncio
async def test_agent_run_steps_api_returns_404_for_missing_run(
    postgres_sessions,
) -> None:
    _, session_factory = postgres_sessions

    run_id = "api-step-missing-run-integration"
    clear_run(session_factory, run_id=run_id)

    client = build_client(session_factory)

    response = client.get(
        f"/api/v1/agents/runs/{run_id}/steps",
    )

    assert response.status_code == 404
    assert response.json()["detail"] == (f"Agent run '{run_id}' was not found.")
