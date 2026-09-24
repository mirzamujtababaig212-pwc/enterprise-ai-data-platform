from __future__ import annotations

from app.control_plane.agent_run_steps.models import (
    AgentRunStep,
    AgentRunStepStatus,
)
from app.control_plane.agent_runs.models import AgentRun
from ai_platform.agents.evaluation.models import AgentRunEvidence
from ai_platform.agents.observability import (
    AgentExecutionEvent,
    AgentExecutionEventType,
)
from tools.models import ToolExecutionFailureCategory

_INVALID_TOOL_CALL_CATEGORIES = frozenset(
    {
        ToolExecutionFailureCategory.SCHEMA_VALIDATION.value,
        ToolExecutionFailureCategory.INVALID_SCHEMA.value,
    }
)


def extract_evidence(
    run: AgentRun,
    steps: list[AgentRunStep],
    events: list[AgentExecutionEvent],
) -> AgentRunEvidence:
    """Build deterministic evaluation evidence from durable agent-run records.

    This function is intentionally read-only. It does not inspect prompts,
    tool arguments, tool outputs, or secrets.
    """
    execution_time_ms = 0.0

    if run.started_at is not None and run.completed_at is not None:
        execution_time_ms = (run.completed_at - run.started_at).total_seconds() * 1000.0

    tool_steps = [step for step in steps if step.tool_name is not None or step.call_id is not None]

    tool_calls_total = len(tool_steps)

    tool_calls_successful = sum(
        1 for step in tool_steps if step.status == AgentRunStepStatus.COMPLETED
    )

    invalid_tool_calls = sum(
        1
        for step in tool_steps
        if (
            step.status == AgentRunStepStatus.FAILED
            and step.failure_category in _INVALID_TOOL_CALL_CATEGORIES
        )
    )

    tool_calls_failed = sum(
        1
        for step in tool_steps
        if (
            step.status == AgentRunStepStatus.FAILED
            and step.failure_category not in _INVALID_TOOL_CALL_CATEGORIES
        )
    )

    governance_denials = sum(
        1
        for event in events
        if (
            event.event_type == AgentExecutionEventType.GOVERNANCE_DECISION
            and event.metadata.get("decision") == "deny"
        )
    )

    return AgentRunEvidence(
        run_id=run.run_id,
        agent_name=run.agent_name,
        agent_version=run.metadata.get("agent_version"),
        tenant_id=run.tenant_id,
        status=run.status.value,
        started_at=run.started_at,
        completed_at=run.completed_at,
        execution_time_ms=execution_time_ms,
        total_steps=len(steps),
        tool_calls_total=tool_calls_total,
        tool_calls_successful=tool_calls_successful,
        tool_calls_failed=tool_calls_failed,
        invalid_tool_calls=invalid_tool_calls,
        governance_denials=governance_denials,
        error_type=run.error_type,
        error_message=run.error_message,
    )
