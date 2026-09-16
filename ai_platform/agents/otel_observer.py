from __future__ import annotations

import asyncio
from dataclasses import dataclass

from opentelemetry import context, trace

from ai_platform.agents.observability import (
    AgentExecutionEvent,
    AgentExecutionEventType,
)
from ai_platform.agents.observer import AgentExecutionObserver


@dataclass
class _ActiveSpan:
    span: trace.Span
    token: object


class OpenTelemetryAgentExecutionObserver(AgentExecutionObserver):
    """
    Records agent execution lifecycle events as OpenTelemetry spans.

    The observer consumes the provider-neutral AgentExecutionEvent contract
    and does not configure an OpenTelemetry TracerProvider or exporter.

    Low-cardinality execution attributes are recorded only. Sensitive
    execution details such as session IDs, user IDs, call IDs, tool
    arguments, tool outputs, governance policies, and arbitrary metadata
    are intentionally excluded.
    """

    def __init__(
        self,
        *,
        tracer: trace.Tracer | None = None,
    ) -> None:
        self._tracer = tracer or trace.get_tracer(
            "ai_platform.agents",
        )
        self._agent_spans: dict[asyncio.Task[object], _ActiveSpan] = {}
        self._llm_spans: dict[asyncio.Task[object], _ActiveSpan] = {}
        self._tool_spans: dict[
            tuple[asyncio.Task[object], str],
            _ActiveSpan,
        ] = {}

    async def record(
        self,
        event: AgentExecutionEvent,
    ) -> None:
        task = asyncio.current_task()

        if task is None:
            raise RuntimeError("OpenTelemetry agent observer requires an active asyncio task.")

        if event.event_type is AgentExecutionEventType.AGENT_STARTED:
            self._start_agent_span(task, event)
            return

        if event.event_type is AgentExecutionEventType.AGENT_COMPLETED:
            self._finish_agent_span(
                task,
                event,
                status=trace.StatusCode.OK,
            )
            return

        if event.event_type is AgentExecutionEventType.AGENT_FAILED:
            self._finish_agent_span(
                task,
                event,
                status=trace.StatusCode.ERROR,
            )
            return

        if event.event_type is AgentExecutionEventType.LLM_REQUESTED:
            self._start_llm_span(task, event)
            return

        if event.event_type is AgentExecutionEventType.LLM_COMPLETED:
            self._finish_llm_span(
                task,
                event,
                status=trace.StatusCode.OK,
            )
            return

        if event.event_type is AgentExecutionEventType.TOOL_CALL_REQUESTED:
            self._start_tool_span(task, event)
            return

        if event.event_type is AgentExecutionEventType.TOOL_CALL_COMPLETED:
            self._finish_tool_span(
                task,
                event,
                status=trace.StatusCode.OK,
            )
            return

        if event.event_type is AgentExecutionEventType.TOOL_CALL_FAILED:
            self._finish_tool_span(
                task,
                event,
                status=trace.StatusCode.ERROR,
            )

    def _start_agent_span(
        self,
        task: asyncio.Task[object],
        event: AgentExecutionEvent,
    ) -> None:
        if task in self._agent_spans:
            return

        span = self._tracer.start_span(
            "agent.run",
        )

        span.set_attribute(
            "agent.name",
            event.agent_name,
        )

        if event.run_id is not None:
            span.set_attribute(
                "agent.run_id",
                event.run_id,
            )

        token = context.attach(
            trace.set_span_in_context(span),
        )

        self._agent_spans[task] = _ActiveSpan(
            span=span,
            token=token,
        )

    def _finish_agent_span(
        self,
        task: asyncio.Task[object],
        event: AgentExecutionEvent,
        *,
        status: trace.StatusCode,
    ) -> None:
        if status is trace.StatusCode.ERROR:
            active_llm = self._llm_spans.pop(task, None)
            if active_llm is not None:
                active_llm.span.set_status(status)
                active_llm.span.end()
                context.detach(active_llm.token)

            active_tools = [key for key in self._tool_spans if key[0] is task]

            for key in active_tools:
                active_tool = self._tool_spans.pop(key)
                active_tool.span.set_status(status)
                active_tool.span.end()
                context.detach(active_tool.token)

        active = self._agent_spans.pop(task, None)

        if active is None:
            return

        active.span.set_status(status)
        active.span.end()

        context.detach(active.token)

    def _start_llm_span(
        self,
        task: asyncio.Task[object],
        event: AgentExecutionEvent,
    ) -> None:
        if task in self._llm_spans:
            return

        span = self._tracer.start_span(
            "agent.llm.request",
        )

        span.set_attribute(
            "agent.name",
            event.agent_name,
        )

        if event.run_id is not None:
            span.set_attribute(
                "agent.run_id",
                event.run_id,
            )

        if event.tool_round is not None:
            span.set_attribute(
                "agent.tool_round",
                event.tool_round,
            )

        if event.provider is not None:
            span.set_attribute(
                "llm.provider",
                event.provider,
            )

        if event.model is not None:
            span.set_attribute(
                "llm.model",
                event.model,
            )

        token = context.attach(
            trace.set_span_in_context(span),
        )

        self._llm_spans[task] = _ActiveSpan(
            span=span,
            token=token,
        )

    def _finish_llm_span(
        self,
        task: asyncio.Task[object],
        event: AgentExecutionEvent,
        *,
        status: trace.StatusCode,
    ) -> None:
        active = self._llm_spans.pop(task, None)

        if active is None:
            return

        if event.provider is not None:
            active.span.set_attribute(
                "llm.provider",
                event.provider,
            )

        if event.model is not None:
            active.span.set_attribute(
                "llm.model",
                event.model,
            )

        active.span.set_status(status)
        active.span.end()

        context.detach(active.token)

    def _start_tool_span(
        self,
        task: asyncio.Task[object],
        event: AgentExecutionEvent,
    ) -> None:
        if event.tool_name is None or event.call_id is None:
            return

        key = (task, event.call_id)

        if key in self._tool_spans:
            return

        span = self._tracer.start_span(
            "agent.tool.call",
        )

        span.set_attribute(
            "agent.name",
            event.agent_name,
        )

        if event.run_id is not None:
            span.set_attribute(
                "agent.run_id",
                event.run_id,
            )

        span.set_attribute(
            "tool.name",
            event.tool_name,
        )

        if event.tool_round is not None:
            span.set_attribute(
                "agent.tool_round",
                event.tool_round,
            )

        token = context.attach(
            trace.set_span_in_context(span),
        )

        self._tool_spans[key] = _ActiveSpan(
            span=span,
            token=token,
        )

    def _finish_tool_span(
        self,
        task: asyncio.Task[object],
        event: AgentExecutionEvent,
        *,
        status: trace.StatusCode,
    ) -> None:
        if event.call_id is None:
            return

        active = self._tool_spans.pop(
            (task, event.call_id),
            None,
        )

        if active is None:
            return

        active.span.set_status(status)
        active.span.end()

        context.detach(active.token)
