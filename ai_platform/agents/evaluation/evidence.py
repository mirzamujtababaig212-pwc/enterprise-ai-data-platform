from __future__ import annotations

from typing import Any

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


def _extract_answer_text(output: Any) -> str | None:
    """Safely extract final answer text from a completed agent run."""
    if output is None:
        return None

    if isinstance(output, str):
        value = output.strip()
        return value or None

    if isinstance(output, dict):
        for key in ("reply", "answer", "text"):
            value = output.get(key)
            if isinstance(value, str):
                value = value.strip()
                if value:
                    return value

    return None


def extract_evidence(
    run: AgentRun,
    steps: list[AgentRunStep],
    events: list[AgentExecutionEvent],
) -> AgentRunEvidence:
    """Build deterministic evaluation evidence from durable agent-run records.

    This function is intentionally read-only. It does not inspect prompts,
    tool arguments, or secrets. RAG step metadata and bounded retrieval
    provenance are evaluated as durable evidence.
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

    answer_text = _extract_answer_text(run.output)
    has_final_answer = answer_text is not None
    final_answer_length = len(answer_text) if answer_text is not None else 0

    rag_queries_total = 0
    rag_sources_retrieved_total = 0
    rag_sources_available_count = 0
    rag_chunk_ids: set[str] = set()

    for step in steps:
        provenance = step.metadata.get("rag_provenance")
        is_rag_step = step.tool_name == "rag.search" or isinstance(provenance, dict)

        if not is_rag_step:
            continue

        rag_queries_total += 1

        if isinstance(provenance, dict):
            retrieved_count = provenance.get("retrieved_count")
            sources = provenance.get("sources")

            if isinstance(retrieved_count, int) and retrieved_count >= 0:
                rag_sources_retrieved_total += retrieved_count
            elif isinstance(sources, (list, tuple)):
                rag_sources_retrieved_total += len(sources)

            if isinstance(sources, (list, tuple)):
                rag_sources_available_count += len(sources)

                for source in sources:
                    if not isinstance(source, dict):
                        continue

                    chunk_id = source.get("chunk_id")
                    if isinstance(chunk_id, str) and chunk_id:
                        rag_chunk_ids.add(chunk_id)

                continue

        output = step.output
        if isinstance(output, dict):
            results = output.get("results")
            if isinstance(results, (list, tuple)):
                rag_sources_retrieved_total += len(results)
                rag_sources_available_count += len(results)

                for result in results:
                    if not isinstance(result, dict):
                        continue

                    chunk_id = result.get("chunk_id")
                    if isinstance(chunk_id, str) and chunk_id:
                        rag_chunk_ids.add(chunk_id)

    rag_unique_chunks_count = len(rag_chunk_ids)

    model_governance = (
        run.request_snapshot.model_governance if run.request_snapshot is not None else None
    )

    effective_model = None
    effective_provider = None
    model_policy_id = None
    model_policy_version = None

    if model_governance is not None:
        effective_model = model_governance.get("effective_model")
        effective_provider = model_governance.get("effective_provider")
        model_policy_id = model_governance.get("policy_id")
        model_policy_version = model_governance.get("policy_version")

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
        rag_queries_total=rag_queries_total,
        rag_sources_retrieved_total=rag_sources_retrieved_total,
        has_final_answer=has_final_answer,
        final_answer_length=final_answer_length,
        final_answer_text=answer_text,
        rag_sources_available_count=rag_sources_available_count,
        rag_unique_chunks_count=rag_unique_chunks_count,
        effective_model=effective_model,
        effective_provider=effective_provider,
        model_policy_id=model_policy_id,
        model_policy_version=model_policy_version,
        error_type=run.error_type,
        error_message=run.error_message,
    )
