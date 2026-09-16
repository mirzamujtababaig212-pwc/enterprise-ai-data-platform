from __future__ import annotations

import asyncio

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from ai_platform.agents.observability import (
    AgentExecutionEvent,
    AgentExecutionEventType,
)
from ai_platform.agents.otel_observer import (
    OpenTelemetryAgentExecutionObserver,
)


def make_observer() -> tuple[
    OpenTelemetryAgentExecutionObserver,
    InMemorySpanExporter,
]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("tests.agents.otel")

    observer = OpenTelemetryAgentExecutionObserver(
        tracer=tracer,
    )

    return observer, exporter


def span_by_name(
    exporter: InMemorySpanExporter,
    name: str,
):
    matches = [span for span in exporter.get_finished_spans() if span.name == name]

    assert len(matches) == 1
    return matches[0]


def run(coro):
    return asyncio.run(coro)


def test_agent_completed_creates_agent_span() -> None:
    observer, exporter = make_observer()

    async def scenario() -> None:
        await observer.record(
            AgentExecutionEvent(
                event_type=AgentExecutionEventType.AGENT_STARTED,
                agent_name="test-agent",
            )
        )
        await observer.record(
            AgentExecutionEvent(
                event_type=AgentExecutionEventType.AGENT_COMPLETED,
                agent_name="test-agent",
            )
        )

    run(scenario())

    span = span_by_name(exporter, "agent.run")

    assert span.status.status_code is trace.StatusCode.OK
    assert span.attributes["agent.name"] == "test-agent"


def test_run_id_is_recorded_on_agent_llm_and_tool_spans() -> None:
    observer, exporter = make_observer()

    async def scenario() -> None:
        await observer.record(
            AgentExecutionEvent(
                event_type=AgentExecutionEventType.AGENT_STARTED,
                agent_name="test-agent",
                run_id="run-123",
            )
        )
        await observer.record(
            AgentExecutionEvent(
                event_type=AgentExecutionEventType.LLM_REQUESTED,
                agent_name="test-agent",
                run_id="run-123",
            )
        )
        await observer.record(
            AgentExecutionEvent(
                event_type=AgentExecutionEventType.LLM_COMPLETED,
                agent_name="test-agent",
                run_id="run-123",
                provider="openai",
                model="gpt-test",
            )
        )
        await observer.record(
            AgentExecutionEvent(
                event_type=AgentExecutionEventType.TOOL_CALL_REQUESTED,
                agent_name="test-agent",
                run_id="run-123",
                tool_name="search_documents",
                call_id="call-123",
            )
        )
        await observer.record(
            AgentExecutionEvent(
                event_type=AgentExecutionEventType.TOOL_CALL_COMPLETED,
                agent_name="test-agent",
                run_id="run-123",
                tool_name="search_documents",
                call_id="call-123",
            )
        )
        await observer.record(
            AgentExecutionEvent(
                event_type=AgentExecutionEventType.AGENT_COMPLETED,
                agent_name="test-agent",
                run_id="run-123",
            )
        )

    run(scenario())

    assert span_by_name(exporter, "agent.run").attributes["agent.run_id"] == "run-123"
    assert span_by_name(exporter, "agent.llm.request").attributes["agent.run_id"] == "run-123"
    assert span_by_name(exporter, "agent.tool.call").attributes["agent.run_id"] == "run-123"


def test_run_id_is_not_recorded_when_absent() -> None:
    observer, exporter = make_observer()

    async def scenario() -> None:
        await observer.record(
            AgentExecutionEvent(
                event_type=AgentExecutionEventType.AGENT_STARTED,
                agent_name="test-agent",
            )
        )
        await observer.record(
            AgentExecutionEvent(
                event_type=AgentExecutionEventType.LLM_REQUESTED,
                agent_name="test-agent",
            )
        )
        await observer.record(
            AgentExecutionEvent(
                event_type=AgentExecutionEventType.LLM_COMPLETED,
                agent_name="test-agent",
            )
        )
        await observer.record(
            AgentExecutionEvent(
                event_type=AgentExecutionEventType.AGENT_COMPLETED,
                agent_name="test-agent",
            )
        )

    run(scenario())

    for span in exporter.get_finished_spans():
        assert "agent.run_id" not in span.attributes


def test_agent_failed_creates_error_agent_span() -> None:
    observer, exporter = make_observer()

    async def scenario() -> None:
        await observer.record(
            AgentExecutionEvent(
                event_type=AgentExecutionEventType.AGENT_STARTED,
                agent_name="test-agent",
            )
        )
        await observer.record(
            AgentExecutionEvent(
                event_type=AgentExecutionEventType.AGENT_FAILED,
                agent_name="test-agent",
                metadata={"error_type": "RuntimeError"},
            )
        )

    run(scenario())

    span = span_by_name(exporter, "agent.run")

    assert span.status.status_code is trace.StatusCode.ERROR
    assert span.attributes["agent.name"] == "test-agent"


def test_llm_span_records_provider_model_and_tool_round() -> None:
    observer, exporter = make_observer()

    async def scenario() -> None:
        await observer.record(
            AgentExecutionEvent(
                event_type=AgentExecutionEventType.AGENT_STARTED,
                agent_name="test-agent",
            )
        )
        await observer.record(
            AgentExecutionEvent(
                event_type=AgentExecutionEventType.LLM_REQUESTED,
                agent_name="test-agent",
                tool_round=2,
            )
        )
        await observer.record(
            AgentExecutionEvent(
                event_type=AgentExecutionEventType.LLM_COMPLETED,
                agent_name="test-agent",
                tool_round=2,
                provider="openai",
                model="gpt-test",
            )
        )
        await observer.record(
            AgentExecutionEvent(
                event_type=AgentExecutionEventType.AGENT_COMPLETED,
                agent_name="test-agent",
            )
        )

    run(scenario())

    span = span_by_name(exporter, "agent.llm.request")

    assert span.status.status_code is trace.StatusCode.OK
    assert span.attributes["agent.name"] == "test-agent"
    assert span.attributes["agent.tool_round"] == 2
    assert span.attributes["llm.provider"] == "openai"
    assert span.attributes["llm.model"] == "gpt-test"


def test_tool_span_records_tool_identity_without_call_id() -> None:
    observer, exporter = make_observer()

    async def scenario() -> None:
        await observer.record(
            AgentExecutionEvent(
                event_type=AgentExecutionEventType.AGENT_STARTED,
                agent_name="test-agent",
            )
        )
        await observer.record(
            AgentExecutionEvent(
                event_type=AgentExecutionEventType.TOOL_CALL_REQUESTED,
                agent_name="test-agent",
                tool_round=1,
                tool_name="search_documents",
                call_id="call-123",
            )
        )
        await observer.record(
            AgentExecutionEvent(
                event_type=AgentExecutionEventType.TOOL_CALL_COMPLETED,
                agent_name="test-agent",
                tool_round=1,
                tool_name="search_documents",
                call_id="call-123",
            )
        )
        await observer.record(
            AgentExecutionEvent(
                event_type=AgentExecutionEventType.AGENT_COMPLETED,
                agent_name="test-agent",
            )
        )

    run(scenario())

    span = span_by_name(exporter, "agent.tool.call")

    assert span.status.status_code is trace.StatusCode.OK
    assert span.attributes["agent.name"] == "test-agent"
    assert span.attributes["tool.name"] == "search_documents"
    assert span.attributes["agent.tool_round"] == 1
    assert "call.id" not in span.attributes
    assert "session.id" not in span.attributes
    assert "user.id" not in span.attributes


def test_failed_tool_span_is_error() -> None:
    observer, exporter = make_observer()

    async def scenario() -> None:
        await observer.record(
            AgentExecutionEvent(
                event_type=AgentExecutionEventType.AGENT_STARTED,
                agent_name="test-agent",
            )
        )
        await observer.record(
            AgentExecutionEvent(
                event_type=AgentExecutionEventType.TOOL_CALL_REQUESTED,
                agent_name="test-agent",
                tool_name="search_documents",
                call_id="call-123",
            )
        )
        await observer.record(
            AgentExecutionEvent(
                event_type=AgentExecutionEventType.TOOL_CALL_FAILED,
                agent_name="test-agent",
                tool_name="search_documents",
                call_id="call-123",
            )
        )
        await observer.record(
            AgentExecutionEvent(
                event_type=AgentExecutionEventType.AGENT_COMPLETED,
                agent_name="test-agent",
            )
        )

    run(scenario())

    span = span_by_name(exporter, "agent.tool.call")

    assert span.status.status_code is trace.StatusCode.ERROR


def test_llm_failure_closes_active_llm_span() -> None:
    observer, exporter = make_observer()

    async def scenario() -> None:
        await observer.record(
            AgentExecutionEvent(
                event_type=AgentExecutionEventType.AGENT_STARTED,
                agent_name="test-agent",
            )
        )
        await observer.record(
            AgentExecutionEvent(
                event_type=AgentExecutionEventType.LLM_REQUESTED,
                agent_name="test-agent",
            )
        )
        await observer.record(
            AgentExecutionEvent(
                event_type=AgentExecutionEventType.AGENT_FAILED,
                agent_name="test-agent",
            )
        )

    run(scenario())

    spans = exporter.get_finished_spans()

    assert {span.name for span in spans} == {
        "agent.llm.request",
        "agent.run",
    }

    assert span_by_name(exporter, "agent.llm.request").status.status_code is trace.StatusCode.ERROR


def test_failed_agent_closes_active_tool_spans() -> None:
    observer, exporter = make_observer()

    async def scenario() -> None:
        await observer.record(
            AgentExecutionEvent(
                event_type=AgentExecutionEventType.AGENT_STARTED,
                agent_name="test-agent",
            )
        )
        await observer.record(
            AgentExecutionEvent(
                event_type=AgentExecutionEventType.TOOL_CALL_REQUESTED,
                agent_name="test-agent",
                tool_name="search_documents",
                call_id="call-123",
            )
        )
        await observer.record(
            AgentExecutionEvent(
                event_type=AgentExecutionEventType.AGENT_FAILED,
                agent_name="test-agent",
            )
        )

    run(scenario())

    spans = exporter.get_finished_spans()

    assert {span.name for span in spans} == {
        "agent.tool.call",
        "agent.run",
    }

    assert span_by_name(exporter, "agent.tool.call").status.status_code is trace.StatusCode.ERROR


def test_sensitive_event_fields_are_not_recorded_as_span_attributes() -> None:
    observer, exporter = make_observer()

    async def scenario() -> None:
        await observer.record(
            AgentExecutionEvent(
                event_type=AgentExecutionEventType.AGENT_STARTED,
                agent_name="test-agent",
                session_id="session-123",
                metadata={
                    "governance_policy": "internal",
                    "secret": "do-not-record",
                },
            )
        )
        await observer.record(
            AgentExecutionEvent(
                event_type=AgentExecutionEventType.LLM_REQUESTED,
                agent_name="test-agent",
                session_id="session-123",
            )
        )
        await observer.record(
            AgentExecutionEvent(
                event_type=AgentExecutionEventType.LLM_COMPLETED,
                agent_name="test-agent",
                session_id="session-123",
                provider="openai",
                model="gpt-test",
                metadata={
                    "prompt_tokens": 100,
                    "secret": "do-not-record",
                },
            )
        )
        await observer.record(
            AgentExecutionEvent(
                event_type=AgentExecutionEventType.AGENT_COMPLETED,
                agent_name="test-agent",
                session_id="session-123",
            )
        )

    run(scenario())

    for span in exporter.get_finished_spans():
        attributes = dict(span.attributes)

        assert "session.id" not in attributes
        assert "user.id" not in attributes
        assert "call.id" not in attributes
        assert "governance.policy" not in attributes
        assert "secret" not in attributes
        assert "prompt_tokens" not in attributes


def test_agent_spans_form_parent_child_trace_hierarchy() -> None:
    observer, exporter = make_observer()

    async def scenario() -> None:
        await observer.record(
            AgentExecutionEvent(
                event_type=AgentExecutionEventType.AGENT_STARTED,
                agent_name="test-agent",
            )
        )
        await observer.record(
            AgentExecutionEvent(
                event_type=AgentExecutionEventType.LLM_REQUESTED,
                agent_name="test-agent",
                tool_round=1,
                provider="openai",
                model="gpt-test",
            )
        )
        await observer.record(
            AgentExecutionEvent(
                event_type=AgentExecutionEventType.LLM_COMPLETED,
                agent_name="test-agent",
                tool_round=1,
                provider="openai",
                model="gpt-test",
            )
        )
        await observer.record(
            AgentExecutionEvent(
                event_type=AgentExecutionEventType.TOOL_CALL_REQUESTED,
                agent_name="test-agent",
                tool_round=1,
                tool_name="search_documents",
                call_id="call-123",
            )
        )
        await observer.record(
            AgentExecutionEvent(
                event_type=AgentExecutionEventType.TOOL_CALL_COMPLETED,
                agent_name="test-agent",
                tool_round=1,
                tool_name="search_documents",
                call_id="call-123",
            )
        )
        await observer.record(
            AgentExecutionEvent(
                event_type=AgentExecutionEventType.AGENT_COMPLETED,
                agent_name="test-agent",
            )
        )

    run(scenario())

    agent_span = span_by_name(exporter, "agent.run")
    llm_span = span_by_name(exporter, "agent.llm.request")
    tool_span = span_by_name(exporter, "agent.tool.call")

    assert agent_span.parent is None

    assert llm_span.parent is not None
    assert llm_span.parent.span_id == agent_span.context.span_id

    assert tool_span.parent is not None
    assert tool_span.parent.span_id == agent_span.context.span_id

    assert llm_span.context.trace_id == agent_span.context.trace_id
    assert tool_span.context.trace_id == agent_span.context.trace_id


def test_irrelevant_agent_events_do_not_create_spans() -> None:
    observer, exporter = make_observer()

    async def scenario() -> None:
        await observer.record(
            AgentExecutionEvent(
                event_type=AgentExecutionEventType.LLM_COMPLETED,
                agent_name="test-agent",
            )
        )

    run(scenario())

    assert not exporter.get_finished_spans()
