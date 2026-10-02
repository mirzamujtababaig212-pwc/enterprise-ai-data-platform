from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from ai_platform.agents.observability import (
    AgentExecutionEvent,
    AgentExecutionEventType,
)

from ai_platform.agents.models import (
    AgentRequest,
    AgentResponse,
)

from app.control_plane.agent_runs.models import (
    AgentRun,
    AgentRunExecutionResult,
    AgentRunStatus,
)

from app.control_plane.agent_run_events.models import AgentRunEventsPage
from app.control_plane.dependencies import get_agent_run_application_service
from app.control_plane.routes.agents import router


class FakeAgentRunApplicationService:
    def __init__(self) -> None:
        self.calls: list[tuple[str, AgentRequest, str | None]] = []
        self.runs = {}
        self.list_calls = []
        self.events = {}
        self.event_list_calls = []

    def get_run(
        self,
        run_id: str,
        *,
        tenant_id: str | None = None,
        principal: str | None = None,
    ):
        del tenant_id, principal
        return self.runs.get(run_id)

    def list_runs(
        self,
        *,
        tenant_id: str | None = None,
        agent_name=None,
        session_id=None,
        user_id=None,
        status=None,
        limit=100,
    ):
        self.list_calls.append(
            {
                "tenant_id": tenant_id,
                "agent_name": agent_name,
                "session_id": session_id,
                "user_id": user_id,
                "status": status,
                "limit": limit,
            }
        )
        return list(self.runs.values())[:limit]

    def list_events(
        self,
        run_id: str,
        *,
        tenant_id: str | None = None,
        principal: str | None = None,
        event_type: AgentExecutionEventType | None = None,
        step_id: str | None = None,
        attempt: int | None = None,
        provider: str | None = None,
        limit=100,
        cursor: str | None = None,
    ):
        del tenant_id, principal

        if run_id not in self.runs:
            raise LookupError(
                f"Agent run '{run_id}' was not found.",
            )

        self.event_list_calls.append(
            {
                "run_id": run_id,
                "event_type": event_type,
                "step_id": step_id,
                "attempt": attempt,
                "provider": provider,
                "limit": limit,
                "cursor": cursor,
            }
        )

        events = list(self.events.get(run_id, []))

        if event_type is not None:
            events = [event for event in events if event.event_type == event_type]

        if step_id is not None:
            events = [event for event in events if event.step_id == step_id]

        if attempt is not None:
            events = [event for event in events if event.attempt == attempt]

        if provider is not None:
            events = [event for event in events if event.provider == provider]

        if cursor is None:
            page_events = events[:limit]
            return AgentRunEventsPage(
                events=page_events,
                next_cursor="cursor-page-2" if len(events) > limit else None,
                has_more=len(events) > limit,
            )

        if cursor == "cursor-page-2":
            page_events = events[limit : limit * 2]
            return AgentRunEventsPage(
                events=page_events,
                next_cursor=None,
                has_more=False,
            )

        raise ValueError(f"Unknown cursor: {cursor}")

    async def execute(
        self,
        *,
        agent_name: str,
        request: AgentRequest,
        idempotency_key: str | None = None,
    ) -> AgentRunExecutionResult:
        self.calls.append(
            (
                agent_name,
                request,
                idempotency_key,
            )
        )

        if agent_name == "missing-agent":
            raise LookupError("Agent 'missing-agent' is not registered.")

        response = AgentResponse(
            agent_name=agent_name,
            output=f"Agent response: {request.input}",
            session_id=request.session_id,
            metadata={
                "provider": "mock",
                "model": "mock-gpt",
                "tool_rounds": 0,
            },
        )

        return AgentRunExecutionResult(
            run_id="run-test-123",
            response=response,
        )


def build_client(
    service: FakeAgentRunApplicationService,
) -> TestClient:
    app = FastAPI()

    @app.middleware("http")
    async def test_identity_middleware(request, call_next):
        request.state.tenant_id = "tenant-acme"
        request.state.principal = "test-principal"
        return await call_next(request)

    app.include_router(router)

    app.dependency_overrides[get_agent_run_application_service] = lambda: service

    return TestClient(app)


def test_run_agent_returns_runtime_response() -> None:
    service = FakeAgentRunApplicationService()
    client = build_client(service)

    response = client.post(
        "/api/v1/agents/enterprise-analyst/run",
        json={
            "input": "Explain RAG.",
            "user_id": "user-123",
            "session_id": "session-123",
            "metadata": {
                "source": "test",
            },
        },
    )

    assert response.status_code == 200

    assert response.json() == {
        "run_id": "run-test-123",
        "agent_name": "enterprise-analyst",
        "output": "Agent response: Explain RAG.",
        "session_id": "session-123",
        "metadata": {
            "provider": "mock",
            "model": "mock-gpt",
            "tool_rounds": 0,
        },
    }

    assert len(service.calls) == 1

    agent_name, request, idempotency_key = service.calls[0]

    assert agent_name == "enterprise-analyst"
    assert request.input == "Explain RAG."
    assert idempotency_key is None
    assert request.user_id == "user-123"
    assert request.session_id == "session-123"
    assert request.metadata == {
        "source": "test",
    }


def test_run_agent_forwards_idempotency_key_header() -> None:
    service = FakeAgentRunApplicationService()
    client = build_client(service)

    response = client.post(
        "/api/v1/agents/enterprise-analyst/run",
        headers={
            "Idempotency-Key": "request-123",
        },
        json={
            "input": "Explain RAG.",
            "user_id": "user-123",
            "session_id": "session-123",
        },
    )

    assert response.status_code == 200
    assert len(service.calls) == 1

    agent_name, request, idempotency_key = service.calls[0]

    assert agent_name == "enterprise-analyst"
    assert request.input == "Explain RAG."
    assert request.user_id == "user-123"
    assert request.session_id == "session-123"
    assert idempotency_key == "request-123"


def test_run_agent_without_idempotency_key_preserves_current_behavior() -> None:
    service = FakeAgentRunApplicationService()
    client = build_client(service)

    response = client.post(
        "/api/v1/agents/enterprise-analyst/run",
        json={
            "input": "Explain RAG.",
            "user_id": "user-123",
        },
    )

    assert response.status_code == 200
    assert len(service.calls) == 1

    _, _, idempotency_key = service.calls[0]

    assert idempotency_key is None


def test_run_agent_maps_idempotency_validation_error_to_422() -> None:
    service = FakeAgentRunApplicationService()

    async def execute_with_error(
        *,
        agent_name: str,
        request: AgentRequest,
        idempotency_key: str | None = None,
    ) -> AgentRunExecutionResult:
        raise ValueError(
            "Idempotency key must not be empty when provided.",
        )

    service.execute = execute_with_error
    client = build_client(service)

    response = client.post(
        "/api/v1/agents/enterprise-analyst/run",
        headers={
            "Idempotency-Key": "   ",
        },
        json={
            "input": "Explain RAG.",
            "user_id": "user-123",
        },
    )

    assert response.status_code == 422
    assert response.json() == {
        "detail": "Idempotency key must not be empty when provided.",
    }


def test_run_agent_maps_idempotency_conflict_to_409() -> None:
    service = FakeAgentRunApplicationService()

    async def execute_with_conflict(
        *,
        agent_name: str,
        request: AgentRequest,
        idempotency_key: str | None = None,
    ) -> AgentRunExecutionResult:
        raise RuntimeError(
            "Idempotency key is already associated with a different request.",
        )

    service.execute = execute_with_conflict
    client = build_client(service)

    response = client.post(
        "/api/v1/agents/enterprise-analyst/run",
        headers={
            "Idempotency-Key": "request-123",
        },
        json={
            "input": "Different request.",
            "user_id": "user-123",
        },
    )

    assert response.status_code == 409
    assert response.json() == {
        "detail": ("Idempotency key is already associated with a different request."),
    }


def test_unknown_agent_returns_404() -> None:
    service = FakeAgentRunApplicationService()
    client = build_client(service)

    response = client.post(
        "/api/v1/agents/missing-agent/run",
        json={
            "input": "Hello.",
        },
    )

    assert response.status_code == 404

    assert response.json() == {"detail": "Agent 'missing-agent' is not registered."}


def test_agent_request_requires_input() -> None:
    service = FakeAgentRunApplicationService()
    client = build_client(service)

    response = client.post(
        "/api/v1/agents/enterprise-analyst/run",
        json={},
    )

    assert response.status_code == 422

    assert service.calls == []


@pytest.mark.asyncio
async def test_agent_runtime_initializes_rag_enabled_agent() -> None:
    from app.control_plane import dependencies

    runtime = await dependencies.get_agent_runtime()

    agents = await runtime._registry.list_agents()

    assert "enterprise-analyst" in {agent.name for agent in agents}
    assert "enterprise-rag-analyst" in {agent.name for agent in agents}

    rag_agent = await runtime._registry.get("enterprise-rag-analyst")

    assert rag_agent is not None
    assert rag_agent.definition.tool_names == (
        "rag.search",
        "vehicle.data.query",
        "agent.delegate",
    )

    tools = await dependencies._tool_registry.list_tools()

    assert any(tool.name == "rag.search" for tool in tools)


def test_list_agent_runs_returns_runs_and_applies_filters() -> None:
    service = FakeAgentRunApplicationService()
    client = build_client(service)

    service.runs["run-1"] = AgentRun(
        run_id="run-1",
        agent_name="enterprise-analyst",
        status=AgentRunStatus.COMPLETED,
        session_id="session-123",
        output="completed",
        metadata={"source": "test"},
    )

    response = client.get(
        "/api/v1/agents/runs",
        params={
            "agent_name": "enterprise-analyst",
            "session_id": "session-123",
            "user_id": "user-123",
            "status": "completed",
            "limit": 25,
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "runs": [
            {
                "run_id": "run-1",
                "agent_name": "enterprise-analyst",
                "status": "completed",
                "session_id": "session-123",
                "started_at": None,
                "completed_at": None,
                "output": "completed",
                "metadata": {"source": "test"},
            }
        ],
    }

    assert service.list_calls == [
        {
            "tenant_id": "tenant-acme",
            "agent_name": "enterprise-analyst",
            "session_id": "session-123",
            "user_id": "user-123",
            "status": AgentRunStatus.COMPLETED,
            "limit": 25,
        }
    ]


def test_list_agent_runs_defaults_to_limit_100() -> None:
    service = FakeAgentRunApplicationService()
    client = build_client(service)

    response = client.get("/api/v1/agents/runs")

    assert response.status_code == 200
    assert response.json() == {"runs": []}
    assert service.list_calls == [
        {
            "tenant_id": "tenant-acme",
            "agent_name": None,
            "session_id": None,
            "user_id": None,
            "status": None,
            "limit": 100,
        }
    ]


@pytest.mark.parametrize(
    "params",
    [
        {"status": "unknown"},
        {"limit": 0},
        {"limit": 101},
    ],
)
def test_list_agent_runs_rejects_invalid_query_parameters(params) -> None:
    service = FakeAgentRunApplicationService()
    client = build_client(service)

    response = client.get(
        "/api/v1/agents/runs",
        params=params,
    )

    assert response.status_code == 422
    assert service.list_calls == []


def test_get_agent_run_returns_detail() -> None:
    service = FakeAgentRunApplicationService()
    client = build_client(service)

    from app.control_plane.agent_runs.models import AgentRun, AgentRunStatus

    service.runs["run-123"] = AgentRun(
        run_id="run-123",
        agent_name="enterprise-analyst",
        status=AgentRunStatus.COMPLETED,
        session_id="session-123",
        output="completed",
        metadata={"source": "test"},
    )

    response = client.get("/api/v1/agents/runs/run-123")

    assert response.status_code == 200
    assert response.json() == {
        "run_id": "run-123",
        "agent_name": "enterprise-analyst",
        "status": "completed",
        "session_id": "session-123",
        "started_at": None,
        "completed_at": None,
        "output": "completed",
        "metadata": {"source": "test"},
    }


@pytest.mark.parametrize(
    "run_status",
    [
        AgentRunStatus.PENDING,
        AgentRunStatus.RUNNING,
        AgentRunStatus.COMPLETED,
        AgentRunStatus.FAILED,
        AgentRunStatus.REJECTED,
    ],
)
def test_get_agent_run_exposes_all_lifecycle_statuses(
    run_status: AgentRunStatus,
) -> None:
    service = FakeAgentRunApplicationService()
    client = build_client(service)

    service.runs["run-status"] = AgentRun(
        run_id="run-status",
        agent_name="enterprise-analyst",
        status=run_status,
        session_id="session-123",
        completed_at=None,
        error_type="RuntimeError" if run_status == AgentRunStatus.FAILED else None,
        error_message="provider failed" if run_status == AgentRunStatus.FAILED else None,
        output=None,
        metadata={"source": "test"},
    )

    response = client.get("/api/v1/agents/runs/run-status")

    assert response.status_code == 200
    assert response.json() == {
        "run_id": "run-status",
        "agent_name": "enterprise-analyst",
        "status": run_status.value,
        "session_id": "session-123",
        "started_at": None,
        "completed_at": None,
        "output": None,
        "metadata": {"source": "test"},
    }

    assert "user_id" not in response.json()
    assert "error_type" not in response.json()
    assert "error_message" not in response.json()


def test_list_agent_runs_serializes_all_lifecycle_statuses() -> None:
    service = FakeAgentRunApplicationService()
    client = build_client(service)

    for index, run_status in enumerate(AgentRunStatus):
        service.runs[f"run-{index}"] = AgentRun(
            run_id=f"run-{index}",
            agent_name="enterprise-analyst",
            status=run_status,
            session_id=f"session-{index}",
            metadata={"status_source": "test"},
        )

    response = client.get("/api/v1/agents/runs")

    assert response.status_code == 200

    runs = response.json()["runs"]

    assert {run["status"] for run in runs} == {
        "pending",
        "running",
        "waiting_for_approval",
        "completed",
        "failed",
        "rejected",
        "cancelled",
    }

    for run in runs:
        assert set(run) == {
            "run_id",
            "agent_name",
            "status",
            "session_id",
            "started_at",
            "completed_at",
            "output",
            "metadata",
        }


def test_get_agent_run_returns_404_when_missing() -> None:
    service = FakeAgentRunApplicationService()
    client = build_client(service)

    response = client.get("/api/v1/agents/runs/missing-run")

    assert response.status_code == 404
    assert response.json() == {
        "detail": "Agent run 'missing-run' was not found.",
    }


def test_list_agent_run_events_returns_events() -> None:
    service = FakeAgentRunApplicationService()
    client = build_client(service)

    service.runs["run-events-123"] = AgentRun(
        run_id="run-events-123",
        agent_name="enterprise-analyst",
        status=AgentRunStatus.COMPLETED,
    )

    service.events["run-events-123"] = [
        AgentExecutionEvent(
            event_type=AgentExecutionEventType.AGENT_STARTED,
            agent_name="enterprise-analyst",
            run_id="run-events-123",
            session_id="session-123",
            user_id="user-123",
            metadata={"source": "test"},
        ),
        AgentExecutionEvent(
            event_type=AgentExecutionEventType.LLM_COMPLETED,
            agent_name="enterprise-analyst",
            run_id="run-events-123",
            session_id="session-123",
            user_id="user-123",
            provider="mock",
            model="mock-gpt",
            metadata={
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        ),
    ]

    response = client.get(
        "/api/v1/agents/runs/run-events-123/events",
        params={"limit": 25},
    )

    assert response.status_code == 200
    assert response.json() == {
        "events": [
            {
                "event_type": "agent.started",
                "agent_name": "enterprise-analyst",
                "run_id": "run-events-123",
                "session_id": "session-123",
                "user_id": "user-123",
                "tool_round": None,
                "tool_name": None,
                "call_id": None,
                "step_id": None,
                "step_index": None,
                "step_name": None,
                "attempt": None,
                "provider": None,
                "model": None,
                "metadata": {"source": "test"},
            },
            {
                "event_type": "llm.completed",
                "agent_name": "enterprise-analyst",
                "run_id": "run-events-123",
                "session_id": "session-123",
                "user_id": "user-123",
                "tool_round": None,
                "tool_name": None,
                "call_id": None,
                "step_id": None,
                "step_index": None,
                "step_name": None,
                "attempt": None,
                "provider": "mock",
                "model": "mock-gpt",
                "metadata": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            },
        ],
        "next_cursor": None,
        "has_more": False,
    }

    assert service.event_list_calls == [
        {
            "run_id": "run-events-123",
            "event_type": None,
            "step_id": None,
            "attempt": None,
            "provider": None,
            "limit": 25,
            "cursor": None,
        }
    ]


def test_list_agent_run_events_supports_cursor_pagination() -> None:
    service = FakeAgentRunApplicationService()
    client = build_client(service)

    service.runs["run-events-pagination"] = AgentRun(
        run_id="run-events-pagination",
        agent_name="enterprise-analyst",
        status=AgentRunStatus.COMPLETED,
    )

    service.events["run-events-pagination"] = [
        AgentExecutionEvent(
            event_type=AgentExecutionEventType.AGENT_STARTED,
            agent_name="enterprise-analyst",
            run_id="run-events-pagination",
            metadata={"sequence": 0},
        ),
        AgentExecutionEvent(
            event_type=AgentExecutionEventType.LLM_REQUESTED,
            agent_name="enterprise-analyst",
            run_id="run-events-pagination",
            metadata={"sequence": 1},
        ),
        AgentExecutionEvent(
            event_type=AgentExecutionEventType.LLM_COMPLETED,
            agent_name="enterprise-analyst",
            run_id="run-events-pagination",
            metadata={"sequence": 2},
        ),
    ]

    first_response = client.get(
        "/api/v1/agents/runs/run-events-pagination/events",
        params={"limit": 2},
    )

    assert first_response.status_code == 200
    first_payload = first_response.json()

    assert len(first_payload["events"]) == 2
    assert first_payload["events"][0]["metadata"]["sequence"] == 0
    assert first_payload["events"][1]["metadata"]["sequence"] == 1
    assert first_payload["has_more"] is True
    assert first_payload["next_cursor"] == "cursor-page-2"

    second_response = client.get(
        "/api/v1/agents/runs/run-events-pagination/events",
        params={
            "limit": 2,
            "cursor": first_payload["next_cursor"],
        },
    )

    assert second_response.status_code == 200
    second_payload = second_response.json()

    assert len(second_payload["events"]) == 1
    assert second_payload["events"][0]["metadata"]["sequence"] == 2
    assert second_payload["has_more"] is False
    assert second_payload["next_cursor"] is None

    assert service.event_list_calls == [
        {
            "run_id": "run-events-pagination",
            "event_type": None,
            "step_id": None,
            "attempt": None,
            "provider": None,
            "limit": 2,
            "cursor": None,
        },
        {
            "run_id": "run-events-pagination",
            "event_type": None,
            "step_id": None,
            "attempt": None,
            "provider": None,
            "limit": 2,
            "cursor": "cursor-page-2",
        },
    ]


def test_list_agent_run_events_returns_empty_for_known_run_without_events() -> None:
    service = FakeAgentRunApplicationService()
    client = build_client(service)

    service.runs["run-empty-events"] = AgentRun(
        run_id="run-empty-events",
        agent_name="enterprise-analyst",
        status=AgentRunStatus.COMPLETED,
    )

    response = client.get(
        "/api/v1/agents/runs/run-empty-events/events",
    )

    assert response.status_code == 200
    assert response.json() == {
        "events": [],
        "next_cursor": None,
        "has_more": False,
    }

    assert service.event_list_calls == [
        {
            "run_id": "run-empty-events",
            "event_type": None,
            "step_id": None,
            "attempt": None,
            "provider": None,
            "limit": 100,
            "cursor": None,
        }
    ]


def test_list_agent_run_events_returns_404_when_run_missing() -> None:
    service = FakeAgentRunApplicationService()
    client = build_client(service)

    response = client.get(
        "/api/v1/agents/runs/missing-run/events",
    )

    assert response.status_code == 404
    assert response.json() == {
        "detail": "Agent run 'missing-run' was not found.",
    }


@pytest.mark.parametrize(
    "params",
    [
        {"limit": 0},
        {"limit": 101},
    ],
)
def test_list_agent_run_events_rejects_invalid_limit(params) -> None:
    service = FakeAgentRunApplicationService()
    client = build_client(service)

    response = client.get(
        "/api/v1/agents/runs/run-123/events",
        params=params,
    )

    assert response.status_code == 422
    assert service.event_list_calls == []
