from __future__ import annotations

from prometheus_client import Counter

from ai_platform.agents.observability import (
    AgentExecutionEvent,
    AgentExecutionEventType,
)
from ai_platform.agents.observer import AgentExecutionObserver

AGENT_EXECUTIONS_TOTAL = Counter(
    "deldai_agent_executions_total",
    "Total agent executions completed or failed.",
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
