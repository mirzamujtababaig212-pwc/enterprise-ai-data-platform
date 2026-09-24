from __future__ import annotations

import pytest

from ai_platform.agents.execution import AgentExecutionContext
from ai_platform.agents.lifecycle import AgentExecutionLifecyclePhase
from ai_platform.agents.llm_messages import (
    assistant_message,
    system_message,
    user_message,
)
from ai_platform.agents.models import (
    AgentDefinition,
    AgentRequest,
    AgentResponse,
)
from ai_platform.agents.observability import AgentExecutionEventType
from ai_platform.agents.registry.in_memory import InMemoryAgentRegistry
from ai_platform.agents.runtime import AgentRuntime
from memory.context.builder import MemoryContext
from tools.authorization.in_memory import InMemoryToolAuthorizer
from tools.authorization.service import ToolAuthorizationService
from tools.execution.service import ToolExecutionService
from tools.models import ToolDefinition
from tools.registry.in_memory import InMemoryToolRegistry


class EmptyMemoryBuilder:
    async def build(
        self,
        namespace: str,
        *,
        query: str | None = None,
        agent_name: str | None = None,
        run_id: str | None = None,
    ) -> MemoryContext:
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
        self.last_context: AgentExecutionContext | None = None

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
        self.last_context = context

        return AgentResponse(
            agent_name=self.definition.name,
            output=self._output,
            session_id=context.session_id,
        )


class RuntimeToolAgent:
    def __init__(self, *, name: str = "runtime-tool-agent") -> None:
        self._definition = AgentDefinition(
            name=name,
            description="Agent used to test runtime tool authorization.",
            system_prompt="You are a tool-enabled test agent.",
            model="test-model",
            tool_names=("test_tool",),
        )
        self.last_tool_results = None

    @property
    def definition(self) -> AgentDefinition:
        return self._definition

    async def run(
        self,
        context: AgentExecutionContext,
    ) -> AgentResponse:
        from ai_platform.agents.execution import AgentToolCall

        self.last_tool_results = await context.execute_tool_calls(
            (
                AgentToolCall(
                    call_id="call-runtime-1",
                    name="test_tool",
                    arguments={"value": 42},
                ),
            )
        )

        return AgentResponse(
            agent_name=self.definition.name,
            output="Tool execution attempted.",
            session_id=context.session_id,
        )


class LifecycleRuntimeAgent:
    def __init__(self) -> None:
        self._definition = AgentDefinition(
            name="lifecycle-runtime-agent",
            description="Lifecycle contract test agent.",
            system_prompt="You are a lifecycle test agent.",
            model="test-model",
        )
        self.phases: list[str] = []
        self.lifecycle_states: list[object] = []

    @property
    def definition(self) -> AgentDefinition:
        return self._definition

    async def prepare_context(self, context: AgentExecutionContext) -> None:
        self.phases.append("prepare_context")
        self.lifecycle_states.append(context.lifecycle_state)

    async def evaluate_pre_execution(
        self,
        context: AgentExecutionContext,
    ) -> None:
        self.phases.append("evaluate_pre_execution")
        self.lifecycle_states.append(context.lifecycle_state)

    async def orchestrate_step(self, context: AgentExecutionContext) -> None:
        self.phases.append("orchestrate_step")
        self.lifecycle_states.append(context.lifecycle_state)

    async def execute_boundary(
        self,
        context: AgentExecutionContext,
    ) -> AgentResponse:
        self.phases.append("execute_boundary")
        self.lifecycle_states.append(context.lifecycle_state)
        return AgentResponse(
            agent_name=self.definition.name,
            output="lifecycle output",
            session_id=context.session_id,
        )

    async def evaluate_post_execution(
        self,
        context: AgentExecutionContext,
        response: AgentResponse,
    ) -> AgentResponse:
        self.phases.append("evaluate_post_execution")
        self.lifecycle_states.append(context.lifecycle_state)
        return AgentResponse(
            agent_name=response.agent_name,
            output=response.output + " post-processed",
            session_id=response.session_id,
            metadata=response.metadata,
        )


class RuntimeTestTool:
    def __init__(self) -> None:
        self._definition = ToolDefinition(
            name="test_tool",
            description="Runtime authorization test tool.",
        )
        self.execution_count = 0

    @property
    def definition(self) -> ToolDefinition:
        return self._definition

    async def execute(self, arguments):
        self.execution_count += 1
        return {
            "status": "success",
            "arguments": arguments,
        }


@pytest.mark.asyncio
async def test_runtime_executes_lifecycle_contract_in_order() -> None:
    registry = InMemoryAgentRegistry()

    agent = LifecycleRuntimeAgent()
    await registry.register(agent)

    runtime = AgentRuntime(registry)

    response = await runtime.run(
        "lifecycle-runtime-agent",
        AgentRequest(input="Run lifecycle."),
    )

    assert agent.phases == [
        "prepare_context",
        "evaluate_pre_execution",
        "orchestrate_step",
        "execute_boundary",
        "evaluate_post_execution",
    ]
    assert response.output == "lifecycle output post-processed"
    assert len(agent.lifecycle_states) == 5
    assert all(state is agent.lifecycle_states[0] for state in agent.lifecycle_states)

    snapshot = await agent.lifecycle_states[0].snapshot()
    assert snapshot.phase is AgentExecutionLifecyclePhase.COMPLETED
    assert [transition.to_phase for transition in snapshot.transitions] == [
        AgentExecutionLifecyclePhase.PREPARING,
        AgentExecutionLifecyclePhase.PRE_EXECUTION,
        AgentExecutionLifecyclePhase.ORCHESTRATING,
        AgentExecutionLifecyclePhase.EXECUTING,
        AgentExecutionLifecyclePhase.POST_EXECUTION,
        AgentExecutionLifecyclePhase.COMPLETED,
    ]


@pytest.mark.asyncio
async def test_runtime_creates_fresh_lifecycle_state_per_run() -> None:
    registry = InMemoryAgentRegistry()

    agent = LifecycleRuntimeAgent()
    await registry.register(agent)

    runtime = AgentRuntime(registry)

    await runtime.run(
        "lifecycle-runtime-agent",
        AgentRequest(input="First lifecycle run."),
    )
    first_state = agent.lifecycle_states[0]

    agent.lifecycle_states.clear()
    agent.phases.clear()

    await runtime.run(
        "lifecycle-runtime-agent",
        AgentRequest(input="Second lifecycle run."),
    )
    second_state = agent.lifecycle_states[0]

    assert first_state is not second_state
    assert len(agent.lifecycle_states) == 5
    assert all(state is second_state for state in agent.lifecycle_states)


@pytest.mark.asyncio
async def test_runtime_stops_lifecycle_when_pre_execution_fails() -> None:
    registry = InMemoryAgentRegistry()

    class FailingLifecycleAgent(LifecycleRuntimeAgent):
        async def evaluate_pre_execution(
            self,
            context: AgentExecutionContext,
        ) -> None:
            self.phases.append("evaluate_pre_execution")
            raise RuntimeError("pre-execution failure")

    agent = FailingLifecycleAgent()
    await registry.register(agent)

    runtime = AgentRuntime(registry)

    with pytest.raises(RuntimeError, match="pre-execution failure"):
        await runtime.run(
            "lifecycle-runtime-agent",
            AgentRequest(input="Run lifecycle."),
        )

    assert agent.phases == [
        "prepare_context",
        "evaluate_pre_execution",
    ]

    assert len(agent.lifecycle_states) == 1

    snapshot = await agent.lifecycle_states[0].snapshot()
    assert snapshot.phase is AgentExecutionLifecyclePhase.FAILED
    assert [transition.to_phase for transition in snapshot.transitions] == [
        AgentExecutionLifecyclePhase.PREPARING,
        AgentExecutionLifecyclePhase.PRE_EXECUTION,
        AgentExecutionLifecyclePhase.FAILED,
    ]


@pytest.mark.asyncio
async def test_runtime_marks_lifecycle_failed_when_execute_boundary_fails() -> None:
    registry = InMemoryAgentRegistry()

    class FailingExecuteAgent(LifecycleRuntimeAgent):
        async def execute_boundary(
            self,
            context: AgentExecutionContext,
        ) -> AgentResponse:
            self.phases.append("execute_boundary")
            self.lifecycle_states.append(context.lifecycle_state)
            raise RuntimeError("execute-boundary failure")

    agent = FailingExecuteAgent()
    await registry.register(agent)

    runtime = AgentRuntime(registry)

    with pytest.raises(RuntimeError, match="execute-boundary failure"):
        await runtime.run(
            "lifecycle-runtime-agent",
            AgentRequest(input="Run lifecycle."),
        )

    assert agent.phases == [
        "prepare_context",
        "evaluate_pre_execution",
        "orchestrate_step",
        "execute_boundary",
    ]
    assert len(agent.lifecycle_states) == 4
    assert all(state is agent.lifecycle_states[0] for state in agent.lifecycle_states)

    snapshot = await agent.lifecycle_states[0].snapshot()
    assert snapshot.phase is AgentExecutionLifecyclePhase.FAILED
    assert [transition.to_phase for transition in snapshot.transitions] == [
        AgentExecutionLifecyclePhase.PREPARING,
        AgentExecutionLifecyclePhase.PRE_EXECUTION,
        AgentExecutionLifecyclePhase.ORCHESTRATING,
        AgentExecutionLifecyclePhase.EXECUTING,
        AgentExecutionLifecyclePhase.FAILED,
    ]


@pytest.mark.asyncio
async def test_runtime_marks_lifecycle_failed_when_post_execution_fails() -> None:
    registry = InMemoryAgentRegistry()

    class FailingPostExecutionAgent(LifecycleRuntimeAgent):
        async def evaluate_post_execution(
            self,
            context: AgentExecutionContext,
            response: AgentResponse,
        ) -> AgentResponse:
            self.phases.append("evaluate_post_execution")
            self.lifecycle_states.append(context.lifecycle_state)
            raise RuntimeError("post-execution failure")

    agent = FailingPostExecutionAgent()
    await registry.register(agent)

    runtime = AgentRuntime(registry)

    with pytest.raises(RuntimeError, match="post-execution failure"):
        await runtime.run(
            "lifecycle-runtime-agent",
            AgentRequest(input="Run lifecycle."),
        )

    assert agent.phases == [
        "prepare_context",
        "evaluate_pre_execution",
        "orchestrate_step",
        "execute_boundary",
        "evaluate_post_execution",
    ]
    assert len(agent.lifecycle_states) == 5
    assert all(state is agent.lifecycle_states[0] for state in agent.lifecycle_states)

    snapshot = await agent.lifecycle_states[0].snapshot()
    assert snapshot.phase is AgentExecutionLifecyclePhase.FAILED
    assert [transition.to_phase for transition in snapshot.transitions] == [
        AgentExecutionLifecyclePhase.PREPARING,
        AgentExecutionLifecyclePhase.PRE_EXECUTION,
        AgentExecutionLifecyclePhase.ORCHESTRATING,
        AgentExecutionLifecyclePhase.EXECUTING,
        AgentExecutionLifecyclePhase.POST_EXECUTION,
        AgentExecutionLifecyclePhase.FAILED,
    ]


@pytest.mark.asyncio
async def test_runtime_injects_orchestration_plan_for_known_agent() -> None:
    registry = InMemoryAgentRegistry()

    agent = FakeAgent(name="enterprise-rag-analyst")
    await registry.register(agent)

    runtime = AgentRuntime(registry)

    await runtime.run(
        "enterprise-rag-analyst",
        AgentRequest(input="Analyze the enterprise vehicle data."),
    )

    assert agent.last_context is not None
    assert agent.last_context.orchestration_plan is not None
    assert [step.step_id for step in agent.last_context.orchestration_state.steps] == [
        "retrieve_evidence",
        "analyze_evidence",
        "produce_answer",
    ]


@pytest.mark.asyncio
async def test_runtime_keeps_orchestration_opt_in_for_unknown_plan() -> None:
    registry = InMemoryAgentRegistry()

    agent = FakeAgent(name="test-agent")
    await registry.register(agent)

    runtime = AgentRuntime(registry)

    await runtime.run(
        "test-agent",
        AgentRequest(input="Run without orchestration."),
    )

    assert agent.last_context is not None
    assert agent.last_context.orchestration_plan is None
    assert agent.last_context.orchestration_state.steps == []
    assert agent.last_context.orchestration_state.current_step_index is None


@pytest.mark.asyncio
async def test_runtime_authorized_principal_can_execute_declared_tool() -> None:
    agent_registry = InMemoryAgentRegistry()
    tool_registry = InMemoryToolRegistry()
    tool = RuntimeTestTool()

    await agent_registry.register(RuntimeToolAgent())
    await tool_registry.register(tool)

    authorizer = InMemoryToolAuthorizer()
    await authorizer.allow(
        "user-authorized",
        "test_tool",
    )

    authorization_service = ToolAuthorizationService(authorizer)
    execution_service = ToolExecutionService(
        tool_registry,
        authorization_service=authorization_service,
    )

    runtime = AgentRuntime(
        agent_registry,
        tool_execution_service=execution_service,
    )

    agent = await agent_registry.get("runtime-tool-agent")

    response = await runtime.run(
        "runtime-tool-agent",
        AgentRequest(
            input="Use the test tool.",
            user_id="user-authorized",
            principal="user-authorized",
        ),
    )

    assert response.output == "Tool execution attempted."
    assert agent.last_tool_results[0].success is True
    assert agent.last_tool_results[0].output == {
        "status": "success",
        "arguments": {"value": 42},
    }
    assert tool.execution_count == 1


@pytest.mark.asyncio
async def test_runtime_unauthorized_principal_cannot_execute_declared_tool() -> None:
    agent_registry = InMemoryAgentRegistry()
    tool_registry = InMemoryToolRegistry()
    tool = RuntimeTestTool()

    await agent_registry.register(RuntimeToolAgent())
    await tool_registry.register(tool)

    authorizer = InMemoryToolAuthorizer()
    authorization_service = ToolAuthorizationService(authorizer)
    execution_service = ToolExecutionService(
        tool_registry,
        authorization_service=authorization_service,
    )

    runtime = AgentRuntime(
        agent_registry,
        tool_execution_service=execution_service,
    )

    agent = await agent_registry.get("runtime-tool-agent")

    response = await runtime.run(
        "runtime-tool-agent",
        AgentRequest(
            input="Use the test tool.",
            user_id="user-unauthorized",
            principal="user-unauthorized",
        ),
    )

    assert response.output == "Tool execution attempted."
    assert agent.last_tool_results[0].success is False
    assert agent.last_tool_results[0].error == ("Tool is not authorized for this principal.")
    assert tool.execution_count == 0


@pytest.mark.asyncio
async def test_runtime_memory_write_emits_started_and_completed_events():
    registry = InMemoryAgentRegistry()

    class MemoryWriteAgent(FakeAgent):
        def __init__(self) -> None:
            super().__init__(name="memory-write-agent", output="remember this")
            self._definition = AgentDefinition(
                name="memory-write-agent",
                description="Memory write test agent.",
                system_prompt="You are a memory test agent.",
                model="test-model",
                memory_write_enabled=True,
                memory_episodic_retention_seconds=3600,
            )

    class TrackingMemoryService:
        def __init__(self) -> None:
            self.calls = []

        async def remember(self, content, **kwargs):
            self.calls.append((content, kwargs))
            return None

    class TrackingObserver:
        def __init__(self) -> None:
            self.events = []

        async def record(self, event) -> None:
            self.events.append(event)

    agent = MemoryWriteAgent()
    memory_service = TrackingMemoryService()
    observer = TrackingObserver()

    await registry.register(agent)

    runtime = AgentRuntime(
        registry,
        memory_context_builder=EmptyMemoryBuilder(),
        memory_service=memory_service,
        observer=observer,
    )

    response = await runtime.run(
        "memory-write-agent",
        AgentRequest(
            input="Generate something worth remembering.",
            session_id="session-secret",
            memory_namespace="tenant-secret",
            metadata={"secret": "must-not-leak"},
        ),
        run_id="run-123",
    )

    assert response.output == "remember this"
    assert memory_service.calls == [
        (
            "remember this",
            {
                "namespace": "tenant-secret",
                "memory_type": "episodic",
                "metadata": {
                    "source": "agent_execution",
                    "agent_name": "memory-write-agent",
                    "session_id": "session-secret",
                },
                "retention_seconds": 3600,
            },
        )
    ]

    assert [event.event_type for event in observer.events] == [
        AgentExecutionEventType.MEMORY_WRITE_STARTED,
        AgentExecutionEventType.MEMORY_WRITE_COMPLETED,
    ]

    started, completed = observer.events

    assert started.agent_name == "memory-write-agent"
    assert started.run_id == "run-123"
    assert started.metadata == {
        "memory_type": "episodic",
    }

    assert completed.agent_name == "memory-write-agent"
    assert completed.run_id == "run-123"
    assert completed.metadata["memory_type"] == "episodic"
    assert completed.metadata["latency_ms"] >= 0.0
    assert "content" not in completed.metadata
    assert "namespace" not in completed.metadata
    assert "session_id" not in completed.metadata
    assert "secret" not in completed.metadata


@pytest.mark.asyncio
async def test_runtime_memory_write_failure_is_suppressed_and_emits_failed_event():
    registry = InMemoryAgentRegistry()

    class MemoryWriteAgent(FakeAgent):
        def __init__(self) -> None:
            super().__init__(name="memory-write-failure-agent", output="remember this")
            self._definition = AgentDefinition(
                name="memory-write-failure-agent",
                description="Memory write failure test agent.",
                system_prompt="You are a memory test agent.",
                model="test-model",
                memory_write_enabled=True,
                memory_episodic_retention_seconds=3600,
            )

    class FailingMemoryService:
        async def remember(self, content, **kwargs):
            raise RuntimeError("secret persistence failure")

    class TrackingObserver:
        def __init__(self) -> None:
            self.events = []

        async def record(self, event) -> None:
            self.events.append(event)

    agent = MemoryWriteAgent()
    observer = TrackingObserver()

    await registry.register(agent)

    runtime = AgentRuntime(
        registry,
        memory_context_builder=EmptyMemoryBuilder(),
        memory_service=FailingMemoryService(),
        observer=observer,
    )

    response = await runtime.run(
        "memory-write-failure-agent",
        AgentRequest(
            input="Generate something worth remembering.",
            memory_namespace="tenant-secret",
            metadata={"secret": "must-not-leak"},
        ),
        run_id="run-456",
    )

    assert response.output == "remember this"

    assert [event.event_type for event in observer.events] == [
        AgentExecutionEventType.MEMORY_WRITE_STARTED,
        AgentExecutionEventType.MEMORY_WRITE_FAILED,
    ]

    started, failed = observer.events

    assert started.metadata == {
        "memory_type": "episodic",
    }

    assert failed.metadata["memory_type"] == "episodic"
    assert failed.metadata["latency_ms"] >= 0.0
    assert failed.metadata["error_type"] == "RuntimeError"

    assert "error" not in failed.metadata
    assert "exception" not in failed.metadata
    assert "content" not in failed.metadata
    assert "namespace" not in failed.metadata
    assert "secret" not in failed.metadata


@pytest.mark.asyncio
async def test_runtime_builds_memory_context_for_requested_namespace():
    registry = InMemoryAgentRegistry()

    agent = FakeAgent()
    await registry.register(agent)

    class TrackingMemoryBuilder:
        def __init__(self) -> None:
            self.requested_namespace = None

        async def build(
            self,
            namespace: str,
            *,
            query: str | None = None,
            agent_name: str | None = None,
            run_id: str | None = None,
        ) -> MemoryContext:
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
async def test_runtime_passes_request_input_to_memory_context_builder():
    registry = InMemoryAgentRegistry()

    agent = FakeAgent()
    await registry.register(agent)

    class TrackingMemoryBuilder:
        def __init__(self) -> None:
            self.requested_namespace = None
            self.requested_query = None

        async def build(
            self,
            namespace: str,
            *,
            query: str | None = None,
            agent_name: str | None = None,
            run_id: str | None = None,
        ) -> MemoryContext:
            self.requested_namespace = namespace
            self.requested_query = query

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
            input="How did we deploy Databricks?",
            memory_namespace="project-a",
        ),
    )

    assert builder.requested_namespace == "project-a"
    assert builder.requested_query == "How did we deploy Databricks?"
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
async def test_runtime_passes_run_id_to_agent() -> None:
    registry = InMemoryAgentRegistry()

    agent = FakeAgent()
    await registry.register(agent)

    runtime = AgentRuntime(
        registry,
    )

    await runtime.run(
        "test-agent",
        AgentRequest(
            input="Trace this execution.",
        ),
        run_id="run-123",
    )

    assert agent.last_context.run_id == "run-123"


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
        memory_episodic_retention_seconds=3600,
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
            retention_seconds: int | None = None,
        ):
            self.calls.append(
                {
                    "content": content,
                    "namespace": namespace,
                    "memory_type": memory_type,
                    "metadata": metadata,
                    "retention_seconds": retention_seconds,
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
            "retention_seconds": 3600,
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


@pytest.mark.asyncio
async def test_runtime_resume_uses_checkpoint_history_without_rebuilding_memory():
    from ai_platform.agents.checkpoint import (
        AgentCheckpointPosition,
        AgentExecutionCheckpoint,
    )
    from ai_platform.agents.llm_messages import user_message

    registry = InMemoryAgentRegistry()

    class RecoverableTestAgent:
        def __init__(self):
            self._definition = AgentDefinition(
                name="recoverable-agent",
                description="Recoverable test agent.",
                system_prompt="You are a recoverable test agent.",
                model="test-model",
            )
            self.last_context = None
            self.last_checkpoint = None

        @property
        def definition(self):
            return self._definition

        async def run(self, context):
            raise AssertionError("run() must not be called during recovery")

        async def resume(self, context, checkpoint):
            self.last_context = context
            self.last_checkpoint = checkpoint

            return AgentResponse(
                agent_name=self.definition.name,
                output="recovered",
                session_id=context.session_id,
            )

    agent = RecoverableTestAgent()
    await registry.register(agent)

    class FailingMemoryBuilder:
        async def build(self, *args, **kwargs):
            raise AssertionError("memory must not be rebuilt during recovery")

    runtime = AgentRuntime(
        registry,
        memory_context_builder=FailingMemoryBuilder(),
    )

    checkpoint = AgentExecutionCheckpoint(
        schema_version=1,
        run_id="run-123",
        agent_name="recoverable-agent",
        session_id="session-123",
        user_id="user-123",
        messages=(user_message("Original request"),),
        tool_round=1,
        position=AgentCheckpointPosition.AFTER_TOOL_EXECUTION,
        metadata={},
    )

    response = await runtime.resume(
        "recoverable-agent",
        AgentRequest(
            input="Original request",
            session_id="session-123",
            user_id="user-123",
            memory_namespace="project-a",
        ),
        checkpoint,
        run_id="run-123",
    )

    assert response.output == "recovered"
    assert agent.last_context.history == checkpoint.messages
    assert agent.last_context.memory is None
    assert agent.last_checkpoint is checkpoint


@pytest.mark.asyncio
async def test_runtime_resume_propagates_execution_ownership_loss_signal():
    import asyncio

    from ai_platform.agents.checkpoint import (
        AgentCheckpointPosition,
        AgentExecutionCheckpoint,
    )
    from ai_platform.agents.llm_messages import user_message

    registry = InMemoryAgentRegistry()

    class RecoverableTestAgent:
        def __init__(self):
            self._definition = AgentDefinition(
                name="recoverable-agent",
                description="Recoverable test agent.",
                system_prompt="You are a recoverable test agent.",
                model="test-model",
            )
            self.last_context = None

        @property
        def definition(self):
            return self._definition

        async def run(self, context):
            raise AssertionError("run() must not be called during recovery")

        async def resume(self, context, checkpoint):
            self.last_context = context
            return AgentResponse(
                agent_name=self.definition.name,
                output="recovered",
                session_id=context.session_id,
            )

    agent = RecoverableTestAgent()
    await registry.register(agent)

    runtime = AgentRuntime(registry)

    checkpoint = AgentExecutionCheckpoint(
        schema_version=1,
        run_id="run-ownership-123",
        agent_name="recoverable-agent",
        session_id="session-123",
        user_id="user-123",
        messages=(user_message("Original request"),),
        tool_round=1,
        position=AgentCheckpointPosition.AFTER_TOOL_EXECUTION,
        metadata={},
    )

    ownership_lost = asyncio.Event()

    response = await runtime.resume(
        "recoverable-agent",
        AgentRequest(
            input="Original request",
            session_id="session-123",
            user_id="user-123",
            memory_namespace="project-a",
        ),
        checkpoint,
        run_id="run-ownership-123",
        execution_ownership_lost=ownership_lost,
    )

    assert response.output == "recovered"
    assert agent.last_context.execution_ownership_lost is ownership_lost


@pytest.mark.asyncio
async def test_runtime_resume_rejects_non_recoverable_agent():
    from ai_platform.agents.checkpoint import (
        AgentCheckpointPosition,
        AgentExecutionCheckpoint,
    )
    from ai_platform.agents.llm_messages import user_message

    registry = InMemoryAgentRegistry()
    agent = FakeAgent(name="non-recoverable-agent")
    await registry.register(agent)

    runtime = AgentRuntime(registry)

    checkpoint = AgentExecutionCheckpoint(
        schema_version=1,
        run_id="run-123",
        agent_name="non-recoverable-agent",
        session_id=None,
        user_id=None,
        messages=(user_message("Original request"),),
        tool_round=1,
        position=AgentCheckpointPosition.AFTER_TOOL_EXECUTION,
        metadata={},
    )

    with pytest.raises(
        TypeError,
        match="does not support durable recovery",
    ):
        await runtime.resume(
            "non-recoverable-agent",
            AgentRequest(input="Original request"),
            checkpoint,
            run_id="run-123",
        )
