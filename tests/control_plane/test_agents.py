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

from app.control_plane.dependencies import get_agent_run_application_service
from app.control_plane.routes.agents import router


class FakeAgentRunApplicationService:
    def __init__(self) -> None:
        self.calls: list[tuple[str, AgentRequest]] = []
        self.runs = {}
        self.list_calls = []
        self.events = {}
        self.event_list_calls = []

    def get_run(self, run_id: str):
        return self.runs.get(run_id)

    def list_runs(
        self,
        *,
        agent_name=None,
        session_id=None,
        user_id=None,
        status=None,
        limit=100,
    ):
        self.list_calls.append(
            {
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
        limit=100,
    ):
        if run_id not in self.runs:
            raise LookupError(
                f"Agent run '{run_id}' was not found.",
            )

        self.event_list_calls.append(
            {
                "run_id": run_id,
                "limit": limit,
            }
        )
        return list(self.events.get(run_id, []))[:limit]

    async def execute(
        self,
        *,
        agent_name: str,
        request: AgentRequest,
    ) -> AgentRunExecutionResult:
        self.calls.append(
            (
                agent_name,
                request,
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

    agent_name, request = service.calls[0]

    assert agent_name == "enterprise-analyst"
    assert request.input == "Explain RAG."
    assert request.user_id == "user-123"
    assert request.session_id == "session-123"
    assert request.metadata == {
        "source": "test",
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
    assert rag_agent.definition.tool_names == ("rag.search",)

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
        "completed",
        "failed",
        "rejected",
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
            metadata={"source": "test"},
        ),
        AgentExecutionEvent(
            event_type=AgentExecutionEventType.LLM_COMPLETED,
            agent_name="enterprise-analyst",
            run_id="run-events-123",
            session_id="session-123",
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
                "tool_round": None,
                "tool_name": None,
                "call_id": None,
                "provider": None,
                "model": None,
                "metadata": {"source": "test"},
            },
            {
                "event_type": "llm.completed",
                "agent_name": "enterprise-analyst",
                "run_id": "run-events-123",
                "session_id": "session-123",
                "tool_round": None,
                "tool_name": None,
                "call_id": None,
                "provider": "mock",
                "model": "mock-gpt",
                "metadata": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            },
        ],
    }

    assert service.event_list_calls == [
        {
            "run_id": "run-events-123",
            "limit": 25,
        }
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
    assert response.json() == {"events": []}

    assert service.event_list_calls == [
        {
            "run_id": "run-empty-events",
            "limit": 100,
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
