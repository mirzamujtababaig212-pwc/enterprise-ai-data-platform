import asyncio

from ai_platform.agents.observability import (
    AgentExecutionEvent,
    AgentExecutionEventType,
)
from ai_platform.agents.prometheus_observer import (
    AGENT_EXECUTIONS_TOTAL,
    AGENT_LLM_REQUESTS_TOTAL,
    AGENT_TOOL_CALLS_TOTAL,
    AGENT_TOOL_FAILURES_TOTAL,
    PrometheusAgentExecutionObserver,
)


def _run(coro):
    return asyncio.run(coro)


def _sample_value(metric, labels):
    for sample in metric.collect()[0].samples:
        if sample.labels == labels:
            return sample.value
    return 0.0


def test_completed_agent_execution_increments_completed_counter():
    observer = PrometheusAgentExecutionObserver()

    before = _sample_value(
        AGENT_EXECUTIONS_TOTAL,
        {
            "agent_name": "test-agent",
            "status": "completed",
        },
    )

    _run(
        observer.record(
            AgentExecutionEvent(
                event_type=AgentExecutionEventType.AGENT_COMPLETED,
                agent_name="test-agent",
                session_id="session-123",
                metadata={"secret": "must-not-be-a-label"},
            )
        )
    )

    after = _sample_value(
        AGENT_EXECUTIONS_TOTAL,
        {
            "agent_name": "test-agent",
            "status": "completed",
        },
    )

    assert after == before + 1


def test_failed_agent_execution_increments_failed_counter():
    observer = PrometheusAgentExecutionObserver()

    before = _sample_value(
        AGENT_EXECUTIONS_TOTAL,
        {
            "agent_name": "test-agent",
            "status": "failed",
        },
    )

    _run(
        observer.record(
            AgentExecutionEvent(
                event_type=AgentExecutionEventType.AGENT_FAILED,
                agent_name="test-agent",
                session_id="session-456",
                metadata={"error_type": "RuntimeError"},
            )
        )
    )

    after = _sample_value(
        AGENT_EXECUTIONS_TOTAL,
        {
            "agent_name": "test-agent",
            "status": "failed",
        },
    )

    assert after == before + 1


def test_llm_request_increments_provider_model_counter():
    observer = PrometheusAgentExecutionObserver()

    before = _sample_value(
        AGENT_LLM_REQUESTS_TOTAL,
        {
            "agent_name": "test-agent",
            "provider": "openai",
            "model": "gpt-4o-mini",
        },
    )

    _run(
        observer.record(
            AgentExecutionEvent(
                event_type=AgentExecutionEventType.LLM_REQUESTED,
                agent_name="test-agent",
                provider="openai",
                model="gpt-4o-mini",
                session_id="session-789",
                metadata={"request_id": "sensitive-request-id"},
            )
        )
    )

    after = _sample_value(
        AGENT_LLM_REQUESTS_TOTAL,
        {
            "agent_name": "test-agent",
            "provider": "openai",
            "model": "gpt-4o-mini",
        },
    )

    assert after == before + 1


def test_llm_request_uses_unknown_for_missing_provider_and_model():
    observer = PrometheusAgentExecutionObserver()

    before = _sample_value(
        AGENT_LLM_REQUESTS_TOTAL,
        {
            "agent_name": "test-agent",
            "provider": "unknown",
            "model": "unknown",
        },
    )

    _run(
        observer.record(
            AgentExecutionEvent(
                event_type=AgentExecutionEventType.LLM_REQUESTED,
                agent_name="test-agent",
            )
        )
    )

    after = _sample_value(
        AGENT_LLM_REQUESTS_TOTAL,
        {
            "agent_name": "test-agent",
            "provider": "unknown",
            "model": "unknown",
        },
    )

    assert after == before + 1


def test_tool_request_increments_tool_counter():
    observer = PrometheusAgentExecutionObserver()

    before = _sample_value(
        AGENT_TOOL_CALLS_TOTAL,
        {
            "agent_name": "test-agent",
            "tool_name": "search_documents",
        },
    )

    _run(
        observer.record(
            AgentExecutionEvent(
                event_type=AgentExecutionEventType.TOOL_CALL_REQUESTED,
                agent_name="test-agent",
                tool_name="search_documents",
                call_id="call-123",
            )
        )
    )

    after = _sample_value(
        AGENT_TOOL_CALLS_TOTAL,
        {
            "agent_name": "test-agent",
            "tool_name": "search_documents",
        },
    )

    assert after == before + 1


def test_failed_tool_call_increments_failure_counter():
    observer = PrometheusAgentExecutionObserver()

    before = _sample_value(
        AGENT_TOOL_FAILURES_TOTAL,
        {
            "agent_name": "test-agent",
            "tool_name": "search_documents",
        },
    )

    _run(
        observer.record(
            AgentExecutionEvent(
                event_type=AgentExecutionEventType.TOOL_CALL_FAILED,
                agent_name="test-agent",
                tool_name="search_documents",
                call_id="call-456",
            )
        )
    )

    after = _sample_value(
        AGENT_TOOL_FAILURES_TOTAL,
        {
            "agent_name": "test-agent",
            "tool_name": "search_documents",
        },
    )

    assert after == before + 1


def test_events_without_required_tool_name_do_not_create_tool_metric():
    observer = PrometheusAgentExecutionObserver()

    before = list(AGENT_TOOL_CALLS_TOTAL.collect()[0].samples)

    _run(
        observer.record(
            AgentExecutionEvent(
                event_type=AgentExecutionEventType.TOOL_CALL_REQUESTED,
                agent_name="test-agent",
            )
        )
    )

    after = list(AGENT_TOOL_CALLS_TOTAL.collect()[0].samples)

    assert after == before


def test_sensitive_execution_fields_are_not_prometheus_labels():
    observer = PrometheusAgentExecutionObserver()

    _run(
        observer.record(
            AgentExecutionEvent(
                event_type=AgentExecutionEventType.TOOL_CALL_REQUESTED,
                agent_name="test-agent",
                session_id="session-secret",
                tool_name="search_documents",
                call_id="call-secret",
                metadata={
                    "governance_policy": "secret-policy",
                    "request_metadata": "secret-request",
                },
            )
        )
    )

    samples = AGENT_TOOL_CALLS_TOTAL.collect()[0].samples

    matching = [
        sample
        for sample in samples
        if sample.labels.get("agent_name") == "test-agent"
        and sample.labels.get("tool_name") == "search_documents"
    ]

    assert matching
    assert set(matching[0].labels) == {
        "agent_name",
        "tool_name",
    }


def test_irrelevant_agent_lifecycle_events_do_not_increment_agent_execution_counter():
    observer = PrometheusAgentExecutionObserver()

    before_completed = _sample_value(
        AGENT_EXECUTIONS_TOTAL,
        {
            "agent_name": "test-agent",
            "status": "completed",
        },
    )
    before_failed = _sample_value(
        AGENT_EXECUTIONS_TOTAL,
        {
            "agent_name": "test-agent",
            "status": "failed",
        },
    )

    _run(
        observer.record(
            AgentExecutionEvent(
                event_type=AgentExecutionEventType.AGENT_STARTED,
                agent_name="test-agent",
            )
        )
    )

    after_completed = _sample_value(
        AGENT_EXECUTIONS_TOTAL,
        {
            "agent_name": "test-agent",
            "status": "completed",
        },
    )
    after_failed = _sample_value(
        AGENT_EXECUTIONS_TOTAL,
        {
            "agent_name": "test-agent",
            "status": "failed",
        },
    )

    assert after_completed == before_completed
    assert after_failed == before_failed
