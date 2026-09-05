from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from ai_platform.agents.models import (
    AgentRequest,
    AgentResponse,
)

from app.control_plane.dependencies import get_agent_runtime
from app.control_plane.routes.agents import router


class FakeAgentRuntime:
    def __init__(self) -> None:
        self.calls: list[tuple[str, AgentRequest]] = []

    async def run(
        self,
        agent_name: str,
        request: AgentRequest,
        *,
        history=(),
    ) -> AgentResponse:
        self.calls.append(
            (
                agent_name,
                request,
            )
        )

        if agent_name == "missing-agent":
            raise LookupError("Agent 'missing-agent' is not registered.")

        return AgentResponse(
            agent_name=agent_name,
            output=f"Agent response: {request.input}",
            session_id=request.session_id,
            metadata={
                "provider": "mock",
                "model": "mock-gpt",
                "tool_rounds": 0,
            },
        )


def build_client(
    runtime: FakeAgentRuntime,
) -> TestClient:
    app = FastAPI()

    app.include_router(router)

    app.dependency_overrides[get_agent_runtime] = lambda: runtime

    return TestClient(app)


def test_run_agent_returns_runtime_response() -> None:
    runtime = FakeAgentRuntime()
    client = build_client(runtime)

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
        "agent_name": "enterprise-analyst",
        "output": "Agent response: Explain RAG.",
        "session_id": "session-123",
        "metadata": {
            "provider": "mock",
            "model": "mock-gpt",
            "tool_rounds": 0,
        },
    }

    assert len(runtime.calls) == 1

    agent_name, request = runtime.calls[0]

    assert agent_name == "enterprise-analyst"
    assert request.input == "Explain RAG."
    assert request.user_id == "user-123"
    assert request.session_id == "session-123"
    assert request.metadata == {
        "source": "test",
    }


def test_unknown_agent_returns_404() -> None:
    runtime = FakeAgentRuntime()
    client = build_client(runtime)

    response = client.post(
        "/api/v1/agents/missing-agent/run",
        json={
            "input": "Hello.",
        },
    )

    assert response.status_code == 404

    assert response.json() == {"detail": "Agent 'missing-agent' is not registered."}


def test_agent_request_requires_input() -> None:
    runtime = FakeAgentRuntime()
    client = build_client(runtime)

    response = client.post(
        "/api/v1/agents/enterprise-analyst/run",
        json={},
    )

    assert response.status_code == 422

    assert runtime.calls == []


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
