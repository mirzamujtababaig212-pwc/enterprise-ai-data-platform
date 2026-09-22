from __future__ import annotations

from prometheus_client import Counter, Histogram

from ai_platform.agents.observability import (
    AgentExecutionEvent,
    AgentExecutionEventType,
)
from ai_platform.agents.observer import AgentExecutionObserver

AGENT_EXECUTIONS_TOTAL = Counter(
    "deldai_agent_executions_total",
    "Total agent executions by terminal status.",
    [
        "agent_name",
        "status",
    ],
)

AGENT_LLM_REQUESTS_TOTAL = Counter(
    "deldai_agent_llm_requests_total",
    "Total LLM requests made by agents.",
    [
        "agent_name",
        "provider",
        "model",
    ],
)

AGENT_TOOL_CALLS_TOTAL = Counter(
    "deldai_agent_tool_calls_total",
    "Total tool calls requested by agents.",
    [
        "agent_name",
        "tool_name",
    ],
)

AGENT_TOOL_FAILURES_TOTAL = Counter(
    "deldai_agent_tool_failures_total",
    "Total tool calls that failed during agent execution.",
    [
        "agent_name",
        "tool_name",
    ],
)

AGENT_MEMORY_RETRIEVALS_TOTAL = Counter(
    "deldai_agent_memory_retrievals_total",
    "Total memory retrieval operations by memory type.",
    [
        "agent_name",
        "memory_type",
    ],
)

AGENT_MEMORY_RETRIEVAL_FAILURES_TOTAL = Counter(
    "deldai_agent_memory_retrieval_failures_total",
    "Total failed memory retrieval operations by memory type.",
    [
        "agent_name",
        "memory_type",
    ],
)

AGENT_MEMORY_RETRIEVAL_ITEMS_TOTAL = Counter(
    "deldai_agent_memory_retrieval_items_total",
    "Total memory items returned by retrieval operations.",
    [
        "agent_name",
        "memory_type",
    ],
)

AGENT_MEMORY_RETRIEVAL_DURATION_SECONDS = Histogram(
    "deldai_agent_memory_retrieval_duration_seconds",
    "Memory retrieval duration in seconds.",
    [
        "agent_name",
        "memory_type",
    ],
)

AGENT_MEMORY_WRITES_TOTAL = Counter(
    "deldai_agent_memory_writes_total",
    "Total memory write operations by memory type.",
    [
        "agent_name",
        "memory_type",
    ],
)

AGENT_MEMORY_WRITE_FAILURES_TOTAL = Counter(
    "deldai_agent_memory_write_failures_total",
    "Total failed memory write operations by memory type.",
    [
        "agent_name",
        "memory_type",
    ],
)

AGENT_MEMORY_WRITE_DURATION_SECONDS = Histogram(
    "deldai_agent_memory_write_duration_seconds",
    "Memory write duration in seconds.",
    [
        "agent_name",
        "memory_type",
    ],
)


class PrometheusAgentExecutionObserver(AgentExecutionObserver):
    """
    Records low-cardinality agent execution events as Prometheus metrics.

    Sensitive execution details such as session IDs, user IDs, call IDs,
    tool arguments, tool outputs, governance policies, and arbitrary event
    metadata are intentionally excluded from Prometheus labels.
    """

    async def record(
        self,
        event: AgentExecutionEvent,
    ) -> None:
        if event.event_type is AgentExecutionEventType.AGENT_COMPLETED:
            AGENT_EXECUTIONS_TOTAL.labels(
                agent_name=event.agent_name,
                status="completed",
            ).inc()
            return

        if event.event_type is AgentExecutionEventType.AGENT_FAILED:
            AGENT_EXECUTIONS_TOTAL.labels(
                agent_name=event.agent_name,
                status="failed",
            ).inc()
            return

        if event.event_type is AgentExecutionEventType.AGENT_CANCELLED:
            AGENT_EXECUTIONS_TOTAL.labels(
                agent_name=event.agent_name,
                status="cancelled",
            ).inc()
            return

        if event.event_type is AgentExecutionEventType.MEMORY_RETRIEVAL_STARTED:
            memory_type = event.metadata.get("memory_type")
            if not isinstance(memory_type, str) or not memory_type:
                return

            AGENT_MEMORY_RETRIEVALS_TOTAL.labels(
                agent_name=event.agent_name,
                memory_type=memory_type,
            ).inc()
            return

        if event.event_type is AgentExecutionEventType.MEMORY_RETRIEVAL_COMPLETED:
            memory_type = event.metadata.get("memory_type")
            if not isinstance(memory_type, str) or not memory_type:
                return

            labels = {
                "agent_name": event.agent_name,
                "memory_type": memory_type,
            }

            returned_count = event.metadata.get("returned_count")
            if isinstance(returned_count, int) and returned_count >= 0:
                AGENT_MEMORY_RETRIEVAL_ITEMS_TOTAL.labels(
                    **labels,
                ).inc(returned_count)

            latency_ms = event.metadata.get("latency_ms")
            if isinstance(latency_ms, (int, float)) and latency_ms >= 0:
                AGENT_MEMORY_RETRIEVAL_DURATION_SECONDS.labels(
                    **labels,
                ).observe(float(latency_ms) / 1000.0)

            return

        if event.event_type is AgentExecutionEventType.MEMORY_RETRIEVAL_FAILED:
            memory_type = event.metadata.get("memory_type")
            if not isinstance(memory_type, str) or not memory_type:
                return

            labels = {
                "agent_name": event.agent_name,
                "memory_type": memory_type,
            }

            AGENT_MEMORY_RETRIEVAL_FAILURES_TOTAL.labels(
                **labels,
            ).inc()

            latency_ms = event.metadata.get("latency_ms")
            if isinstance(latency_ms, (int, float)) and latency_ms >= 0:
                AGENT_MEMORY_RETRIEVAL_DURATION_SECONDS.labels(
                    **labels,
                ).observe(float(latency_ms) / 1000.0)

            return

        if event.event_type is AgentExecutionEventType.MEMORY_WRITE_STARTED:
            memory_type = event.metadata.get("memory_type")
            if not isinstance(memory_type, str) or not memory_type:
                return

            AGENT_MEMORY_WRITES_TOTAL.labels(
                agent_name=event.agent_name,
                memory_type=memory_type,
            ).inc()
            return

        if event.event_type is AgentExecutionEventType.MEMORY_WRITE_COMPLETED:
            memory_type = event.metadata.get("memory_type")
            if not isinstance(memory_type, str) or not memory_type:
                return

            latency_ms = event.metadata.get("latency_ms")
            if isinstance(latency_ms, (int, float)) and latency_ms >= 0:
                AGENT_MEMORY_WRITE_DURATION_SECONDS.labels(
                    agent_name=event.agent_name,
                    memory_type=memory_type,
                ).observe(float(latency_ms) / 1000.0)

            return

        if event.event_type is AgentExecutionEventType.MEMORY_WRITE_FAILED:
            memory_type = event.metadata.get("memory_type")
            if not isinstance(memory_type, str) or not memory_type:
                return

            labels = {
                "agent_name": event.agent_name,
                "memory_type": memory_type,
            }

            AGENT_MEMORY_WRITE_FAILURES_TOTAL.labels(
                **labels,
            ).inc()

            latency_ms = event.metadata.get("latency_ms")
            if isinstance(latency_ms, (int, float)) and latency_ms >= 0:
                AGENT_MEMORY_WRITE_DURATION_SECONDS.labels(
                    **labels,
                ).observe(float(latency_ms) / 1000.0)

            return

        if event.event_type is AgentExecutionEventType.LLM_REQUESTED:
            AGENT_LLM_REQUESTS_TOTAL.labels(
                agent_name=event.agent_name,
                provider=event.provider or "unknown",
                model=event.model or "unknown",
            ).inc()
            return

        if event.event_type is AgentExecutionEventType.TOOL_CALL_REQUESTED:
            if event.tool_name is None:
                return

            AGENT_TOOL_CALLS_TOTAL.labels(
                agent_name=event.agent_name,
                tool_name=event.tool_name,
            ).inc()
            return

        if event.event_type is AgentExecutionEventType.TOOL_CALL_FAILED:
            if event.tool_name is None:
                return

            AGENT_TOOL_FAILURES_TOTAL.labels(
                agent_name=event.agent_name,
                tool_name=event.tool_name,
            ).inc()
