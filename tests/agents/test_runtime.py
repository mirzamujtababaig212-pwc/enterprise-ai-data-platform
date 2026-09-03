from __future__ import annotations

import pytest

from ai_platform.agents.execution import AgentExecutionContext
from ai_platform.agents.models import (
    AgentDefinition,
    AgentRequest,
    AgentResponse,
)
from ai_platform.agents.registry.in_memory import InMemoryAgentRegistry
from ai_platform.agents.runtime import AgentRuntime
from ai_platform.agents.llm_messages import (
    assistant_message,
    system_message,
    user_message,
)


class FakeAgent:
    def __init__(
        self,
        name: str = "test-agent",
        *,
        enabled: bool = True,
        output: str = "test output",
    ) -> None:
        self._definition = AgentDefinition(
            name=name,
            description="Test agent.",
            system_prompt="You are a test agent.",
            model="test-model",
            enabled=enabled,
        )
        self._output = output
        self.run_count = 0
        self.last_request: AgentRequest | None = None
        self.last_history = ()

    @property
    def definition(self) -> AgentDefinition:
        return self._definition

    async def run(
        self,
        context: AgentExecutionContext,
    ) -> AgentResponse:
        self.run_count += 1
        self.last_request = context.request
        self.last_history = context.history

        return AgentResponse(
            agent_name=self.definition.name,
            output=self._output,
            session_id=context.session_id,
        )


@pytest.mark.asyncio
async def test_runtime_executes_registered_agent():
    registry = InMemoryAgentRegistry()

    agent = FakeAgent(
        name="data-engineering-agent",
        output="Pipeline looks healthy.",
    )

    await registry.register(agent)

    runtime = AgentRuntime(
        registry,
    )

    response = await runtime.run(
        "data-engineering-agent",
        AgentRequest(
            input="Check the pipeline.",
            session_id="session-123",
        ),
    )

    assert response.agent_name == ("data-engineering-agent")
    assert response.output == ("Pipeline looks healthy.")
    assert response.session_id == "session-123"


@pytest.mark.asyncio
async def test_runtime_passes_request_to_agent():
    registry = InMemoryAgentRegistry()

    agent = FakeAgent()

    await registry.register(agent)

    runtime = AgentRuntime(
        registry,
    )

    request = AgentRequest(
        input="Find ingestion failures.",
        session_id="session-456",
        user_id="user-789",
        metadata={
            "source": "api",
        },
    )

    await runtime.run(
        "test-agent",
        request,
    )

    assert agent.run_count == 1
    assert agent.last_request is request


@pytest.mark.asyncio
async def test_runtime_passes_history_to_agent() -> None:
    registry = InMemoryAgentRegistry()

    agent = FakeAgent()

    await registry.register(agent)

    runtime = AgentRuntime(
        registry,
    )

    history = (
        user_message("What is RAG?"),
        assistant_message("RAG retrieves relevant context."),
    )

    await runtime.run(
        "test-agent",
        AgentRequest(
            input="Why is retrieval useful?",
            session_id="session-123",
        ),
        history=history,
    )

    assert agent.last_history == history


@pytest.mark.asyncio
async def test_runtime_rejects_invalid_history() -> None:
    registry = InMemoryAgentRegistry()

    agent = FakeAgent()

    await registry.register(agent)

    runtime = AgentRuntime(
        registry,
    )

    with pytest.raises(
        TypeError,
        match="Agent execution history must contain AgentMessage instances",
    ):
        await runtime.run(
            "test-agent",
            AgentRequest(
                input="Hello.",
            ),
            history=("invalid",),  # type: ignore[arg-type]
        )

    assert agent.run_count == 0


@pytest.mark.asyncio
async def test_runtime_executes_agent_only_once():
    registry = InMemoryAgentRegistry()

    agent = FakeAgent()

    await registry.register(agent)

    runtime = AgentRuntime(
        registry,
    )

    await runtime.run(
        "test-agent",
        AgentRequest(
            input="Hello.",
        ),
    )

    assert agent.run_count == 1


@pytest.mark.asyncio
async def test_runtime_raises_for_unknown_agent():
    registry = InMemoryAgentRegistry()

    runtime = AgentRuntime(
        registry,
    )

    with pytest.raises(
        LookupError,
        match="Agent 'unknown-agent' is not registered",
    ):
        await runtime.run(
            "unknown-agent",
            AgentRequest(
                input="Hello.",
            ),
        )


@pytest.mark.asyncio
async def test_runtime_rejects_empty_agent_name():
    registry = InMemoryAgentRegistry()

    runtime = AgentRuntime(
        registry,
    )

    with pytest.raises(
        ValueError,
        match="Agent name must not be empty",
    ):
        await runtime.run(
            "",
            AgentRequest(
                input="Hello.",
            ),
        )


@pytest.mark.asyncio
async def test_runtime_rejects_whitespace_agent_name():
    registry = InMemoryAgentRegistry()

    runtime = AgentRuntime(
        registry,
    )

    with pytest.raises(
        ValueError,
        match="Agent name must not be empty",
    ):
        await runtime.run(
            "   ",
            AgentRequest(
                input="Hello.",
            ),
        )


@pytest.mark.asyncio
async def test_runtime_does_not_execute_disabled_agent():
    registry = InMemoryAgentRegistry()

    agent = FakeAgent(
        name="disabled-agent",
        enabled=False,
    )

    await registry.register(agent)

    runtime = AgentRuntime(
        registry,
    )

    with pytest.raises(
        RuntimeError,
        match="Agent 'disabled-agent' is disabled",
    ):
        await runtime.run(
            "disabled-agent",
            AgentRequest(
                input="Hello.",
            ),
        )

    assert agent.run_count == 0


@pytest.mark.asyncio
async def test_runtime_returns_agent_response_unchanged():
    registry = InMemoryAgentRegistry()

    class ResponseAgent:
        @property
        def definition(self) -> AgentDefinition:
            return AgentDefinition(
                name="response-agent",
                description="Response agent.",
                system_prompt="You return responses.",
            )

        async def run(
            self,
            context: AgentExecutionContext,
        ) -> AgentResponse:
            return AgentResponse(
                agent_name="response-agent",
                output={
                    "answer": "exact response",
                    "items": [
                        1,
                        2,
                        3,
                    ],
                },
                session_id=context.session_id,
                metadata={
                    "custom": "value",
                },
            )

    agent = ResponseAgent()

    await registry.register(agent)

    runtime = AgentRuntime(
        registry,
    )

    response = await runtime.run(
        "response-agent",
        AgentRequest(
            input="Return something.",
            session_id="session-999",
        ),
    )

    assert response.output == {
        "answer": "exact response",
        "items": [
            1,
            2,
            3,
        ],
    }

    assert response.metadata == {
        "custom": "value",
    }

    assert response.session_id == "session-999"


@pytest.mark.asyncio
async def test_runtime_uses_registry_lookup():
    class TrackingRegistry:
        def __init__(
            self,
            agent: FakeAgent,
        ) -> None:
            self.agent = agent
            self.requested_name: str | None = None

        async def get(
            self,
            name: str,
        ) -> FakeAgent | None:
            self.requested_name = name
            return self.agent

    agent = FakeAgent(
        name="tracked-agent",
    )

    registry = TrackingRegistry(
        agent,
    )

    runtime = AgentRuntime(
        registry,
    )

    await runtime.run(
        "tracked-agent",
        AgentRequest(
            input="Hello.",
        ),
    )

    assert registry.requested_name == ("tracked-agent")


@pytest.mark.asyncio
async def test_runtime_does_not_depend_on_agent_implementation_details():
    registry = InMemoryAgentRegistry()

    agent = FakeAgent(
        name="minimal-agent",
        output="minimal response",
    )

    await registry.register(agent)

    runtime = AgentRuntime(
        registry,
    )

    response = await runtime.run(
        "minimal-agent",
        AgentRequest(
            input="Hello.",
        ),
    )

    assert response.output == "minimal response"


@pytest.mark.asyncio
async def test_runtime_supports_sessionless_requests():
    registry = InMemoryAgentRegistry()

    agent = FakeAgent(
        name="sessionless-agent",
        output="sessionless response",
    )

    await registry.register(agent)

    runtime = AgentRuntime(
        registry,
    )

    response = await runtime.run(
        "sessionless-agent",
        AgentRequest(
            input="Hello.",
        ),
    )

    assert response.session_id is None
    assert response.output == "sessionless response"


@pytest.mark.asyncio
async def test_runtime_propagates_history_into_llm_execution_context() -> None:
    registry = InMemoryAgentRegistry()

    class HistoryAgent:
        @property
        def definition(self) -> AgentDefinition:
            return AgentDefinition(
                name="history-agent",
                description="History-aware agent.",
                system_prompt="You are a history-aware agent.",
                model="mock-gpt",
            )

        async def run(
            self,
            context: AgentExecutionContext,
        ) -> AgentResponse:
            messages = context.build_llm_messages()

            assert messages == (
                system_message("You are a history-aware agent."),
                user_message("What is RAG?"),
                assistant_message("RAG retrieves relevant context."),
                user_message("Why is retrieval useful?"),
            )

            return AgentResponse(
                agent_name=self.definition.name,
                output="History received.",
                session_id=context.session_id,
            )

    agent = HistoryAgent()

    await registry.register(agent)

    runtime = AgentRuntime(
        registry,
    )

    history = (
        user_message("What is RAG?"),
        assistant_message("RAG retrieves relevant context."),
    )

    response = await runtime.run(
        "history-agent",
        AgentRequest(
            input="Why is retrieval useful?",
        ),
        history=history,
    )

    assert response.output == "History received."
