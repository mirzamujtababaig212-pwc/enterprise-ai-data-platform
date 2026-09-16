from __future__ import annotations

import pytest

from ai_platform.agents.execution import AgentExecutionContext
from memory.context.builder import MemoryContext
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


class EmptyMemoryBuilder:
    async def build(self, namespace: str) -> MemoryContext:
        return MemoryContext(
            working=(),
            semantic=(),
            episodic=(),
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
        self.last_memory = None

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
        self.last_memory = context.memory

        return AgentResponse(
            agent_name=self.definition.name,
            output=self._output,
            session_id=context.session_id,
        )


@pytest.mark.asyncio
async def test_runtime_builds_memory_context_for_requested_namespace():
    registry = InMemoryAgentRegistry()

    agent = FakeAgent()
    await registry.register(agent)

    class TrackingMemoryBuilder:
        def __init__(self) -> None:
            self.requested_namespace = None

        async def build(self, namespace: str) -> MemoryContext:
            self.requested_namespace = namespace
            return MemoryContext(
                working=(),
                semantic=(),
                episodic=(),
            )

    builder = TrackingMemoryBuilder()

    runtime = AgentRuntime(
        registry,
        memory_context_builder=builder,
    )

    await runtime.run(
        "test-agent",
        AgentRequest(
            input="Use memory.",
            memory_namespace="project-a",
        ),
    )

    assert builder.requested_namespace == "project-a"
    assert isinstance(agent.last_memory, MemoryContext)
    assert agent.last_memory.is_empty


@pytest.mark.asyncio
async def test_runtime_rejects_memory_without_configured_builder():
    registry = InMemoryAgentRegistry()

    agent = FakeAgent()
    await registry.register(agent)

    runtime = AgentRuntime(
        registry,
    )

    with pytest.raises(
        RuntimeError,
        match="requested memory but no MemoryContextBuilder is configured",
    ):
        await runtime.run(
            "test-agent",
            AgentRequest(
                input="Use memory.",
                memory_namespace="project-a",
            ),
        )

    assert agent.run_count == 0


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


@pytest.mark.asyncio
async def test_runtime_writes_successful_string_response_to_episodic_memory():
    registry = InMemoryAgentRegistry()

    agent = FakeAgent(
        name="memory-writing-agent",
        output="The customer prefers Snowflake.",
    )
    agent._definition = AgentDefinition(
        name="memory-writing-agent",
        description="Memory-writing agent.",
        system_prompt="You are a memory-writing agent.",
        model="test-model",
        memory_write_enabled=True,
    )

    await registry.register(agent)

    class TrackingMemoryService:
        def __init__(self) -> None:
            self.calls = []

        async def remember(
            self,
            content: str,
            *,
            namespace: str,
            memory_type: str,
            metadata: dict | None = None,
        ):
            self.calls.append(
                {
                    "content": content,
                    "namespace": namespace,
                    "memory_type": memory_type,
                    "metadata": metadata,
                }
            )

    memory_service = TrackingMemoryService()

    runtime = AgentRuntime(
        registry,
        memory_context_builder=EmptyMemoryBuilder(),
        memory_service=memory_service,
    )

    response = await runtime.run(
        "memory-writing-agent",
        AgentRequest(
            input="What does the customer prefer?",
            session_id="session-123",
            memory_namespace="customer-456",
        ),
    )

    assert response.output == "The customer prefers Snowflake."
    assert memory_service.calls == [
        {
            "content": "The customer prefers Snowflake.",
            "namespace": "customer-456",
            "memory_type": "episodic",
            "metadata": {
                "source": "agent_execution",
                "agent_name": "memory-writing-agent",
                "session_id": "session-123",
            },
        }
    ]


@pytest.mark.asyncio
async def test_runtime_does_not_write_memory_when_write_back_is_disabled():
    registry = InMemoryAgentRegistry()

    agent = FakeAgent(
        name="memory-disabled-agent",
        output="Do not persist this.",
    )
    await registry.register(agent)

    class TrackingMemoryService:
        def __init__(self) -> None:
            self.calls = 0

        async def remember(self, *args, **kwargs):
            self.calls += 1

    memory_service = TrackingMemoryService()

    runtime = AgentRuntime(
        registry,
        memory_context_builder=EmptyMemoryBuilder(),
        memory_service=memory_service,
    )

    response = await runtime.run(
        "memory-disabled-agent",
        AgentRequest(
            input="Hello.",
            memory_namespace="project-a",
        ),
    )

    assert response.output == "Do not persist this."
    assert memory_service.calls == 0


@pytest.mark.asyncio
async def test_runtime_does_not_write_memory_without_namespace():
    registry = InMemoryAgentRegistry()

    agent = FakeAgent(
        name="memory-no-namespace-agent",
        output="This has nowhere to go.",
    )
    agent._definition = AgentDefinition(
        name="memory-no-namespace-agent",
        description="Memory agent without namespace.",
        system_prompt="You are a memory agent.",
        model="test-model",
        memory_write_enabled=True,
    )
    await registry.register(agent)

    class TrackingMemoryService:
        def __init__(self) -> None:
            self.calls = 0

        async def remember(self, *args, **kwargs):
            self.calls += 1

    memory_service = TrackingMemoryService()

    runtime = AgentRuntime(
        registry,
        memory_service=memory_service,
    )

    response = await runtime.run(
        "memory-no-namespace-agent",
        AgentRequest(
            input="Hello.",
        ),
    )

    assert response.output == "This has nowhere to go."
    assert memory_service.calls == 0


@pytest.mark.asyncio
async def test_runtime_does_not_write_non_string_response():
    registry = InMemoryAgentRegistry()

    class StructuredAgent:
        @property
        def definition(self) -> AgentDefinition:
            return AgentDefinition(
                name="structured-memory-agent",
                description="Structured memory agent.",
                system_prompt="You return structured responses.",
                memory_write_enabled=True,
            )

        async def run(
            self,
            context: AgentExecutionContext,
        ) -> AgentResponse:
            return AgentResponse(
                agent_name=self.definition.name,
                output={
                    "answer": "Do not serialize me.",
                    "confidence": 0.9,
                },
            )

    agent = StructuredAgent()
    await registry.register(agent)

    class TrackingMemoryService:
        def __init__(self) -> None:
            self.calls = 0

        async def remember(self, *args, **kwargs):
            self.calls += 1

    memory_service = TrackingMemoryService()

    runtime = AgentRuntime(
        registry,
        memory_context_builder=EmptyMemoryBuilder(),
        memory_service=memory_service,
    )

    response = await runtime.run(
        "structured-memory-agent",
        AgentRequest(
            input="Return structured data.",
            memory_namespace="project-a",
        ),
    )

    assert response.output == {
        "answer": "Do not serialize me.",
        "confidence": 0.9,
    }
    assert memory_service.calls == 0


@pytest.mark.asyncio
async def test_runtime_returns_success_when_memory_write_fails():
    registry = InMemoryAgentRegistry()

    agent = FakeAgent(
        name="memory-failing-agent",
        output="The execution succeeded.",
    )
    agent._definition = AgentDefinition(
        name="memory-failing-agent",
        description="Memory failure test agent.",
        system_prompt="You are a test agent.",
        memory_write_enabled=True,
    )
    await registry.register(agent)

    class FailingMemoryService:
        async def remember(self, *args, **kwargs):
            raise RuntimeError("memory backend unavailable")

    runtime = AgentRuntime(
        registry,
        memory_context_builder=EmptyMemoryBuilder(),
        memory_service=FailingMemoryService(),
    )

    response = await runtime.run(
        "memory-failing-agent",
        AgentRequest(
            input="Run successfully.",
            memory_namespace="project-a",
        ),
    )

    assert response.output == "The execution succeeded."
