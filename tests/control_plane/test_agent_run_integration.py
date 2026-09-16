from __future__ import annotations

import asyncio

from fastapi import FastAPI
from fastapi.testclient import TestClient

from ai_platform.agents.execution import AgentExecutionContext
from ai_platform.agents.models import AgentDefinition, AgentResponse
from ai_platform.agents.observability import (
    AgentExecutionEvent,
    AgentExecutionEventType,
)
from ai_platform.agents.registry.in_memory import InMemoryAgentRegistry
from ai_platform.agents.runtime import AgentRuntime
from app.control_plane.agent_runs.application_service import (
    AgentRunApplicationService,
)
from app.control_plane.agent_runs.in_memory import InMemoryAgentRunRepository
from app.control_plane.dependencies import get_agent_run_application_service
from app.control_plane.routes.agents import router


class RecordingObserver:
    def __init__(self) -> None:
        self.events: list[AgentExecutionEvent] = []

    async def record(self, event: AgentExecutionEvent) -> None:
        self.events.append(event)


class DeterministicAgent:
    def __init__(self, observer: RecordingObserver) -> None:
        self._definition = AgentDefinition(
            name="test-agent",
            description="Deterministic integration-test agent.",
            system_prompt="You are a deterministic test agent.",
            model="test-model",
            enabled=True,
        )
        self._observer = observer

    @property
    def definition(self) -> AgentDefinition:
        return self._definition

    async def run(
        self,
        context: AgentExecutionContext,
    ) -> AgentResponse:
        await self._observer.record(
            AgentExecutionEvent(
                event_type=AgentExecutionEventType.AGENT_STARTED,
                agent_name=self.definition.name,
                run_id=context.run_id,
                session_id=context.session_id,
            )
        )

        return AgentResponse(
            agent_name=self.definition.name,
            output="deterministic integration response",
            session_id=context.session_id,
        )


def build_integration_client(
    service: AgentRunApplicationService,
) -> TestClient:
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_agent_run_application_service] = lambda: service
    return TestClient(app)


def test_agent_run_id_propagates_from_http_to_persistence_and_event() -> None:
    registry = InMemoryAgentRegistry()
    observer = RecordingObserver()
    agent = DeterministicAgent(observer)

    asyncio.run(registry.register(agent))

    runtime = AgentRuntime(registry)
    repository = InMemoryAgentRunRepository()

    service = AgentRunApplicationService(
        runtime=runtime,
        repository=repository,
    )

    client = build_integration_client(service)

    response = client.post(
        "/api/v1/agents/test-agent/run",
        json={
            "input": "Trace this execution.",
            "user_id": "user-123",
            "session_id": "session-123",
            "metadata": {
                "source": "integration-test",
            },
        },
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["agent_name"] == "test-agent"
    assert payload["output"] == "deterministic integration response"
    assert payload["session_id"] == "session-123"

    response_run_id = payload["run_id"]

    persisted_run = repository.get(response_run_id)

    assert persisted_run is not None
    assert persisted_run.run_id == response_run_id
    assert persisted_run.agent_name == "test-agent"
    assert persisted_run.session_id == "session-123"
    assert persisted_run.user_id == "user-123"
    assert persisted_run.status.value == "completed"
    assert persisted_run.output == "deterministic integration response"

    assert len(observer.events) == 1

    event = observer.events[0]

    assert event.event_type == AgentExecutionEventType.AGENT_STARTED
    assert event.agent_name == "test-agent"
    assert event.run_id == response_run_id
    assert event.session_id == "session-123"
