from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest

from ai_platform.agents.llm_agent import LLMAgent
from ai_platform.agents.models import AgentDefinition, AgentRequest
from ai_platform.agents.observability import (
    AgentExecutionEvent,
    AgentExecutionEventType,
)
from ai_platform.agents.otel_observer import (
    OpenTelemetryAgentExecutionObserver,
)
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from ai_platform.agents.registry.in_memory import InMemoryAgentRegistry
from ai_platform.agents.runtime import AgentRuntime
from ai_platform.agents.tool_calls import AgentToolCall
from tools.authorization.in_memory import InMemoryToolAuthorizer
from tools.authorization.policy import MetadataAuthorizationPolicy
from tools.authorization.service import ToolAuthorizationService
from tools.execution.service import ToolExecutionService
from app.control_plane.agent_run_events.tool_authorization_observer import (
    ToolAuthorizationAuditObserver,
)
from tools.mcp.config import MCPServerConfig
from tools.mcp.manager import MCPServerManager
from tools.registry.in_memory import InMemoryToolRegistry
from ai_platform.llm_gateway.routing.router import Router
from ai_platform.llm_gateway.exceptions.provider_exceptions import (
    ProviderConnectionError,
)
from ai_platform.llm_gateway.routing.fallback_executor import FallbackExecutor

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "tools" / "mcp" / "fixtures"
SEARCH_SERVER = FIXTURES_DIR / "test_server.py"


class FakeAgentExecutionObserver:
    def __init__(self) -> None:
        self.events: list[AgentExecutionEvent] = []

    async def record(
        self,
        event: AgentExecutionEvent,
    ) -> None:
        self.events.append(event)


class FakeToolCallingLLMGateway:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []

    async def route_chat(
        self,
        request: dict[str, Any],
    ) -> dict[str, Any]:
        self.requests.append(request)

        if len(self.requests) == 1:
            return {
                "provider": "fake",
                "model": request["model"],
                "reply": "",
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
                "tool_calls": [
                    AgentToolCall(
                        call_id="call-observe-1",
                        name="search_documents",
                        arguments={
                            "query": "enterprise AI",
                        },
                    ),
                ],
            }

        return {
            "provider": "fake",
            "model": request["model"],
            "reply": "Enterprise AI platform architecture.",
            "usage": {
                "prompt_tokens": 25,
                "completion_tokens": 10,
                "total_tokens": 35,
            },
        }


def make_agent(
    observer: FakeAgentExecutionObserver,
) -> LLMAgent:
    definition = AgentDefinition(
        name="observable-mcp-agent",
        description="Agent integration test for observability.",
        system_prompt=(
            "You are an enterprise AI agent. " "Use the search_documents MCP tool when required."
        ),
        model="mock-gpt",
        tool_names=("search_documents",),
    )

    return LLMAgent(
        definition,
        observer=observer,
    )


def make_authorization_service(
    expected_server: str,
) -> tuple[InMemoryToolAuthorizer, ToolAuthorizationService]:
    policy = MetadataAuthorizationPolicy(
        {
            "source": "mcp",
            "mcp_server": expected_server,
        }
    )

    authorizer = InMemoryToolAuthorizer(
        policy=policy,
    )

    return authorizer, ToolAuthorizationService(authorizer)


async def make_runtime(
    *,
    observer: FakeAgentExecutionObserver,
    tool_registry: InMemoryToolRegistry,
    authorization_service: ToolAuthorizationService,
    gateway: FakeToolCallingLLMGateway,
) -> AgentRuntime:
    agent = make_agent(observer)

    agent_registry = InMemoryAgentRegistry()
    await agent_registry.register(agent)

    execution_service = ToolExecutionService(
        tool_registry,
        authorization_service=authorization_service,
        audit_sink=ToolAuthorizationAuditObserver(observer),
    )

    return AgentRuntime(
        agent_registry,
        tool_execution_service=execution_service,
        llm_gateway=gateway,
    )


@pytest.mark.asyncio
async def test_agent_mcp_execution_emits_complete_observability_lifecycle() -> None:
    tool_registry = InMemoryToolRegistry()

    authorizer, authorization_service = make_authorization_service(
        "document-server",
    )

    manager = MCPServerManager(tool_registry)

    config = MCPServerConfig(
        name="document-server",
        transport="stdio",
        command=sys.executable,
        args=(str(SEARCH_SERVER),),
    )

    await manager.register_server(config)

    observer = FakeAgentExecutionObserver()
    gateway = FakeToolCallingLLMGateway()

    runtime = await make_runtime(
        observer=observer,
        tool_registry=tool_registry,
        authorization_service=authorization_service,
        gateway=gateway,
    )

    await authorizer.allow(
        "user-observe-123",
        "search_documents",
    )

    try:
        definitions = await manager.connect_and_discover(
            "document-server",
        )

        assert definitions[0].metadata == {
            "source": "mcp",
            "mcp_server": "document-server",
            "capability": "unclassified",
            "risk_tier": "unknown",
            "side_effect": True,
        }

        response = await runtime.run(
            "observable-mcp-agent",
            AgentRequest(
                input="Find enterprise AI architecture information.",
                session_id="session-observe-123",
                user_id="user-observe-123",
                principal="user-observe-123",
                metadata={
                    "source": "agent-observability-test",
                },
            ),
        )

        assert response.output == "Enterprise AI platform architecture."

        assert [event.event_type for event in observer.events] == [
            AgentExecutionEventType.AGENT_STARTED,
            AgentExecutionEventType.LLM_REQUESTED,
            AgentExecutionEventType.LLM_COMPLETED,
            AgentExecutionEventType.TOOL_CALL_REQUESTED,
            AgentExecutionEventType.TOOL_AUTHORIZATION_DECISION,
            AgentExecutionEventType.TOOL_CALL_COMPLETED,
            AgentExecutionEventType.LLM_REQUESTED,
            AgentExecutionEventType.LLM_COMPLETED,
            AgentExecutionEventType.RUNTIME_DECISION,
            AgentExecutionEventType.AGENT_COMPLETED,
        ]

        started = observer.events[0]
        first_llm = observer.events[2]
        tool_requested = observer.events[3]
        authorization = observer.events[4]
        tool_completed = observer.events[5]
        final_llm = observer.events[7]
        completed = observer.events[9]

        assert started.agent_name == "observable-mcp-agent"
        assert started.session_id == "session-observe-123"

        assert first_llm.tool_round == 0
        assert first_llm.provider == "fake"
        assert first_llm.model == "mock-gpt"
        assert first_llm.metadata == {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        }

        assert tool_requested.agent_name == "observable-mcp-agent"
        assert tool_requested.session_id == "session-observe-123"
        assert tool_requested.tool_round == 1
        assert tool_requested.tool_name == "search_documents"
        assert tool_requested.call_id == "call-observe-1"
        assert tool_requested.metadata == {}

        assert authorization.event_type is AgentExecutionEventType.TOOL_AUTHORIZATION_DECISION
        assert authorization.agent_name == "observable-mcp-agent"
        assert authorization.session_id == "session-observe-123"
        assert authorization.tool_name == "search_documents"
        assert authorization.call_id == "call-observe-1"
        assert authorization.metadata == {
            "allowed": True,
            "reason": "Tool is authorized.",
            "policy_id": "metadata_policy",
            "policy_version": "1.0",
        }
        assert "principal" not in authorization.metadata
        assert "user-observe-123" not in authorization.metadata

        assert tool_completed.agent_name == "observable-mcp-agent"
        assert tool_completed.session_id == "session-observe-123"
        assert tool_completed.tool_round == 1
        assert tool_completed.tool_name == "search_documents"
        assert tool_completed.call_id == "call-observe-1"
        assert tool_completed.metadata == {
            "execution_provenance": {
                "tool_source": "mcp",
                "mcp_server": "document-server",
                "tool_provider": {
                    "kind": "mcp",
                    "name": "document-server",
                },
                "authorization_decision": True,
                "authorization_policy_id": "metadata_policy",
                "authorization_policy_version": "1.0",
                "execution_status": "completed",
            },
        }

        provenance = tool_completed.metadata["execution_provenance"]
        assert "arguments" not in provenance
        assert "output" not in provenance
        assert "principal" not in provenance
        assert "sensitive" not in provenance

        assert final_llm.tool_round == 1
        assert final_llm.provider == "fake"
        assert final_llm.model == "mock-gpt"
        assert final_llm.metadata == {
            "prompt_tokens": 25,
            "completion_tokens": 10,
            "total_tokens": 35,
        }

        assert completed.tool_round == 1
        assert completed.provider == "fake"
        assert completed.model == "mock-gpt"

        for event in observer.events:
            assert "arguments" not in event.metadata
            assert "output" not in event.metadata
            assert "result" not in event.metadata
            assert "governance_policy" not in event.metadata
            assert "request_metadata" not in event.metadata

        assert len(gateway.requests) == 2

    finally:
        await manager.disconnect_all()


@pytest.mark.asyncio
async def test_agent_mcp_execution_creates_otel_trace_hierarchy() -> None:
    tool_registry = InMemoryToolRegistry()

    authorizer, authorization_service = make_authorization_service(
        "document-server",
    )

    manager = MCPServerManager(tool_registry)

    config = MCPServerConfig(
        name="document-server",
        transport="stdio",
        command=sys.executable,
        args=(str(SEARCH_SERVER),),
    )

    await manager.register_server(config)

    exporter = InMemorySpanExporter()
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = tracer_provider.get_tracer("tests.integration.agent")

    observer = OpenTelemetryAgentExecutionObserver(
        tracer=tracer,
    )
    gateway = FakeToolCallingLLMGateway()

    agent = make_agent(observer)

    agent_registry = InMemoryAgentRegistry()
    await agent_registry.register(agent)

    tool_registry = InMemoryToolRegistry()
    execution_service = ToolExecutionService(
        tool_registry,
        authorization_service=authorization_service,
    )

    runtime = AgentRuntime(
        agent_registry,
        tool_execution_service=execution_service,
        tool_registry=tool_registry,
        llm_gateway=gateway,
    )

    await authorizer.allow(
        "user-otel-integration",
        "search_documents",
    )

    try:
        definitions = await manager.connect_and_discover(
            "document-server",
        )

        assert definitions[0].metadata == {
            "source": "mcp",
            "mcp_server": "document-server",
            "capability": "unclassified",
            "risk_tier": "unknown",
            "side_effect": True,
        }

        response = await runtime.run(
            "observable-mcp-agent",
            AgentRequest(
                input="Find enterprise AI architecture information.",
                session_id="session-otel-integration",
                user_id="user-otel-integration",
            ),
        )

        assert response.output == "Enterprise AI platform architecture."

        spans = exporter.get_finished_spans()
        assert {span.name for span in spans} == {
            "agent.run",
            "agent.llm.request",
            "agent.tool.call",
        }

        agent_span = next(span for span in spans if span.name == "agent.run")
        llm_span = next(span for span in spans if span.name == "agent.llm.request")
        tool_span = next(span for span in spans if span.name == "agent.tool.call")

        assert agent_span.parent is None

        assert llm_span.parent is not None
        assert llm_span.parent.span_id == agent_span.context.span_id

        assert tool_span.parent is not None
        assert tool_span.parent.span_id == agent_span.context.span_id

        assert llm_span.context.trace_id == agent_span.context.trace_id
        assert tool_span.context.trace_id == agent_span.context.trace_id

        assert agent_span.attributes["agent.name"] == "observable-mcp-agent"

        assert llm_span.attributes["agent.name"] == "observable-mcp-agent"
        assert llm_span.attributes["llm.provider"] == "fake"
        assert llm_span.attributes["llm.model"] == "mock-gpt"

        assert tool_span.attributes["agent.name"] == "observable-mcp-agent"
        assert tool_span.attributes["tool.name"] == "search_documents"

        assert "session.id" not in agent_span.attributes
        assert "user.id" not in agent_span.attributes
        assert "call.id" not in tool_span.attributes

        assert len(gateway.requests) == 2

    finally:
        await manager.disconnect_all()


@pytest.mark.asyncio
async def test_agent_llm_gateway_creates_nested_otel_trace_hierarchy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify agent LLM spans contain the real Gateway spans."""

    exporter = InMemorySpanExporter()
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = tracer_provider.get_tracer("tests.integration.agent.gateway")
    monkeypatch.setattr(
        "ai_platform.llm_gateway.routing.router.tracer",
        tracer,
    )

    observer = OpenTelemetryAgentExecutionObserver(
        tracer=tracer,
    )

    agent = make_agent(observer)

    agent_registry = InMemoryAgentRegistry()
    await agent_registry.register(agent)

    tool_registry = InMemoryToolRegistry()

    runtime = AgentRuntime(
        agent_registry,
        tool_registry=tool_registry,
        llm_gateway=Router(),
    )

    response = await runtime.run(
        "observable-mcp-agent",
        AgentRequest(
            input="Explain enterprise AI architecture.",
            session_id="session-gateway-otel",
            user_id="user-gateway-otel",
        ),
    )

    assert response.output == "Mock response: Explain enterprise AI architecture."

    spans = exporter.get_finished_spans()

    assert {span.name for span in spans} == {
        "agent.run",
        "agent.llm.request",
        "gateway.chat",
        "provider_call",
    }

    agent_span = next(span for span in spans if span.name == "agent.run")
    llm_span = next(span for span in spans if span.name == "agent.llm.request")
    gateway_span = next(span for span in spans if span.name == "gateway.chat")
    provider_span = next(span for span in spans if span.name == "provider_call")

    assert agent_span.parent is None

    assert llm_span.parent is not None
    assert llm_span.parent.span_id == agent_span.context.span_id

    assert gateway_span.parent is not None
    assert gateway_span.parent.span_id == llm_span.context.span_id

    assert provider_span.parent is not None
    assert provider_span.parent.span_id == gateway_span.context.span_id

    assert llm_span.context.trace_id == agent_span.context.trace_id
    assert gateway_span.context.trace_id == agent_span.context.trace_id
    assert provider_span.context.trace_id == agent_span.context.trace_id

    assert agent_span.attributes["agent.name"] == "observable-mcp-agent"

    assert llm_span.attributes["agent.name"] == "observable-mcp-agent"
    assert llm_span.attributes["llm.provider"] == "mock"
    assert llm_span.attributes["llm.model"] == "mock-gpt"

    assert gateway_span.attributes["llm.model"] == "mock-gpt"
    assert "llm.requested_provider" not in gateway_span.attributes
    assert gateway_span.attributes["llm.provider"] == "mock"
    assert gateway_span.attributes["llm.attempt_count"] == 1

    assert "session.id" not in agent_span.attributes
    assert "user.id" not in agent_span.attributes
    assert "session.id" not in llm_span.attributes
    assert "user.id" not in llm_span.attributes

    assert "prompt" not in gateway_span.attributes
    assert "response" not in gateway_span.attributes
    assert "user.id" not in gateway_span.attributes


@pytest.mark.asyncio
async def test_agent_llm_gateway_provider_failure_marks_full_otel_hierarchy_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify provider failures propagate ERROR status through the full trace."""

    exporter = InMemorySpanExporter()
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = tracer_provider.get_tracer("tests.integration.agent.gateway.error")

    monkeypatch.setattr(
        "ai_platform.llm_gateway.routing.router.tracer",
        tracer,
    )

    observer = OpenTelemetryAgentExecutionObserver(
        tracer=tracer,
    )

    agent = make_agent(observer)

    agent_registry = InMemoryAgentRegistry()
    await agent_registry.register(agent)

    tool_registry = InMemoryToolRegistry()

    provider_a = Mock()
    provider_a.name = "provider-a"
    provider_a.chat = AsyncMock(
        side_effect=ProviderConnectionError("provider-a failure"),
    )

    provider_b = Mock()
    provider_b.name = "provider-b"
    provider_b.chat = AsyncMock(
        side_effect=ProviderConnectionError("provider-b failure"),
    )

    fallback_executor = FallbackExecutor(max_retries=0)

    router = Router(
        fallback_executor=fallback_executor,
    )

    monkeypatch.setattr(
        router.routing_resolver,
        "is_logical_model",
        Mock(return_value=False),
    )
    monkeypatch.setattr(
        router.routing_resolver,
        "resolve",
        Mock(return_value=[provider_a, provider_b]),
    )

    runtime = AgentRuntime(
        agent_registry,
        tool_registry=tool_registry,
        llm_gateway=router,
    )

    with pytest.raises(
        ProviderConnectionError,
        match="provider-b failure",
    ):
        await runtime.run(
            "observable-mcp-agent",
            AgentRequest(
                input="Explain enterprise AI architecture.",
                session_id="session-gateway-error",
                user_id="user-gateway-error",
            ),
        )

    spans = exporter.get_finished_spans()

    assert {span.name for span in spans} == {
        "agent.run",
        "agent.llm.request",
        "gateway.chat",
        "provider_call",
    }

    agent_span = next(span for span in spans if span.name == "agent.run")
    llm_span = next(span for span in spans if span.name == "agent.llm.request")
    gateway_span = next(span for span in spans if span.name == "gateway.chat")
    provider_spans = [span for span in spans if span.name == "provider_call"]

    assert len(provider_spans) == 2

    assert agent_span.parent is None

    assert llm_span.parent is not None
    assert llm_span.parent.span_id == agent_span.context.span_id

    assert gateway_span.parent is not None
    assert gateway_span.parent.span_id == llm_span.context.span_id

    for provider_span in provider_spans:
        assert provider_span.parent is not None
        assert provider_span.parent.span_id == gateway_span.context.span_id
        assert provider_span.context.trace_id == agent_span.context.trace_id
        assert provider_span.context.trace_id == llm_span.context.trace_id
        assert provider_span.context.trace_id == gateway_span.context.trace_id
        assert provider_span.status.status_code is trace.StatusCode.ERROR
        assert provider_span.attributes["provider.model"] == "mock-gpt"
        assert "prompt" not in provider_span.attributes
        assert "response" not in provider_span.attributes
        assert "user.id" not in provider_span.attributes
        assert "session.id" not in provider_span.attributes

    assert agent_span.context.trace_id == llm_span.context.trace_id
    assert agent_span.context.trace_id == gateway_span.context.trace_id

    assert gateway_span.status.status_code is trace.StatusCode.ERROR
    assert llm_span.status.status_code is trace.StatusCode.ERROR
    assert agent_span.status.status_code is trace.StatusCode.ERROR

    assert gateway_span.attributes["llm.model"] == "mock-gpt"

    assert "prompt" not in gateway_span.attributes
    assert "response" not in gateway_span.attributes
    assert "user.id" not in gateway_span.attributes
    assert "session.id" not in gateway_span.attributes

    provider_names = {span.attributes["provider.name"] for span in provider_spans}
    assert provider_names == {
        "provider-a",
        "provider-b",
    }


@pytest.mark.asyncio
async def test_agent_mcp_authorization_denial_emits_tool_failure_event() -> None:
    tool_registry = InMemoryToolRegistry()

    authorizer, authorization_service = make_authorization_service(
        "finance-server",
    )

    manager = MCPServerManager(tool_registry)

    config = MCPServerConfig(
        name="document-server",
        transport="stdio",
        command=sys.executable,
        args=(str(SEARCH_SERVER),),
    )

    await manager.register_server(config)

    observer = FakeAgentExecutionObserver()
    gateway = FakeToolCallingLLMGateway()

    runtime = await make_runtime(
        observer=observer,
        tool_registry=tool_registry,
        authorization_service=authorization_service,
        gateway=gateway,
    )

    await authorizer.allow(
        "user-observe-denied",
        "search_documents",
    )

    try:
        definitions = await manager.connect_and_discover(
            "document-server",
        )

        assert definitions[0].metadata == {
            "source": "mcp",
            "mcp_server": "document-server",
            "capability": "unclassified",
            "risk_tier": "unknown",
            "side_effect": True,
        }

        response = await runtime.run(
            "observable-mcp-agent",
            AgentRequest(
                input="Find enterprise AI architecture information.",
                session_id="session-observe-denied",
                user_id="user-observe-denied",
                principal="user-observe-denied",
            ),
        )

        assert response.output == "Enterprise AI platform architecture."

        tool_events = [
            event
            for event in observer.events
            if event.event_type
            in {
                AgentExecutionEventType.TOOL_CALL_REQUESTED,
                AgentExecutionEventType.TOOL_CALL_COMPLETED,
                AgentExecutionEventType.TOOL_CALL_FAILED,
            }
        ]

        assert [event.event_type for event in tool_events] == [
            AgentExecutionEventType.TOOL_CALL_REQUESTED,
            AgentExecutionEventType.TOOL_CALL_FAILED,
        ]

        failed = tool_events[-1]

        assert failed.agent_name == "observable-mcp-agent"
        assert failed.session_id == "session-observe-denied"
        assert failed.tool_round == 1
        assert failed.tool_name == "search_documents"
        assert failed.call_id == "call-observe-1"
        assert failed.metadata == {
            "failure_category": "authorization",
        }

        assert all(
            event.event_type != AgentExecutionEventType.TOOL_CALL_COMPLETED for event in tool_events
        )

        authorization_events = [
            event
            for event in observer.events
            if event.event_type is AgentExecutionEventType.TOOL_AUTHORIZATION_DECISION
        ]

        assert len(authorization_events) == 1

        authorization = authorization_events[0]

        assert authorization.agent_name == "observable-mcp-agent"
        assert authorization.session_id == "session-observe-denied"
        assert authorization.tool_round is None
        assert authorization.tool_name == "search_documents"
        assert authorization.call_id == "call-observe-1"
        assert authorization.metadata == {
            "allowed": False,
            "reason": "Authorization metadata requirement failed: mcp_server='finance-server'.",
            "policy_id": "metadata_policy",
            "policy_version": "1.0",
        }
        assert "principal" not in authorization.metadata
        assert "user-observe-denied" not in authorization.metadata

        assert [event.event_type for event in observer.events] == [
            AgentExecutionEventType.AGENT_STARTED,
            AgentExecutionEventType.LLM_REQUESTED,
            AgentExecutionEventType.LLM_COMPLETED,
            AgentExecutionEventType.TOOL_CALL_REQUESTED,
            AgentExecutionEventType.TOOL_AUTHORIZATION_DECISION,
            AgentExecutionEventType.TOOL_CALL_FAILED,
            AgentExecutionEventType.LLM_REQUESTED,
            AgentExecutionEventType.LLM_COMPLETED,
            AgentExecutionEventType.RUNTIME_DECISION,
            AgentExecutionEventType.AGENT_COMPLETED,
        ]

        for event in observer.events:
            assert event.metadata.get("error") is None
            assert "arguments" not in event.metadata
            assert "output" not in event.metadata
            assert "result" not in event.metadata

        assert len(gateway.requests) == 2

    finally:
        await manager.disconnect_all()
