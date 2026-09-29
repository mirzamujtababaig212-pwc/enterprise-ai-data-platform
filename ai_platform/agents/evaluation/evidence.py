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


def _extract_context_diagnostics(
    events: list[AgentExecutionEvent],
) -> tuple[int, int, int, int, bool, int, dict[str, int]]:
    """Aggregate bounded context-assembly diagnostics from durable events."""
    context_assembly_events_total = 0
    context_messages_total = 0
    context_estimated_tokens_total = 0
    context_estimated_tokens_max = 0
    context_budget_exceeded = False
    context_source_profile_changes = 0
    context_source_counts: dict[str, int] = {}
    previous_source_profile: frozenset[str] | None = None

    for event in events:
        if event.event_type != AgentExecutionEventType.CONTEXT_ASSEMBLY_COMPLETED:
            continue

        context_assembly_events_total += 1
        metadata = event.metadata

        total_messages = metadata.get("total_messages")
        if isinstance(total_messages, int) and not isinstance(total_messages, bool):
            context_messages_total += max(0, total_messages)

        estimated_tokens = metadata.get("estimated_tokens")
        if isinstance(estimated_tokens, int) and not isinstance(estimated_tokens, bool):
            estimated_tokens = max(0, estimated_tokens)
            context_estimated_tokens_total += estimated_tokens
            context_estimated_tokens_max = max(
                context_estimated_tokens_max,
                estimated_tokens,
            )

        source_counts = metadata.get("source_counts")
        if isinstance(source_counts, dict):
            current_source_profile = frozenset(
                source_type
                for source_type, count in source_counts.items()
                if isinstance(source_type, str)
                and isinstance(count, int)
                and not isinstance(count, bool)
                and count > 0
            )

            if (
                previous_source_profile is not None
                and current_source_profile != previous_source_profile
            ):
                context_source_profile_changes += 1

            previous_source_profile = current_source_profile

            for source_type, count in source_counts.items():
                if not isinstance(source_type, str):
                    continue
                if not isinstance(count, int) or isinstance(count, bool):
                    continue
                context_source_counts[source_type] = context_source_counts.get(
                    source_type, 0
                ) + max(0, count)

        budget_status = metadata.get("budget_status")
        if isinstance(budget_status, dict):
            within_budget = budget_status.get("within_budget")
            if within_budget is False:
                context_budget_exceeded = True

    return (
        context_assembly_events_total,
        context_messages_total,
        context_estimated_tokens_total,
        context_estimated_tokens_max,
        context_budget_exceeded,
        context_source_profile_changes,
        context_source_counts,
    )


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

    retrieval_scores: list[float] = []
    reranker_scores: list[float] = []

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

                    retrieval_score = source.get("retrieval_score")
                    if isinstance(retrieval_score, (int, float)) and not isinstance(
                        retrieval_score, bool
                    ):
                        retrieval_scores.append(float(retrieval_score))
                    else:
                        # Compatibility with older provenance records.
                        score = source.get("score")
                        if isinstance(score, (int, float)) and not isinstance(score, bool):
                            retrieval_scores.append(float(score))

                    reranker_score = source.get("reranker_score")
                    if isinstance(reranker_score, (int, float)) and not isinstance(
                        reranker_score, bool
                    ):
                        reranker_scores.append(float(reranker_score))

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

                    retrieval_score = result.get("retrieval_score")
                    if isinstance(retrieval_score, (int, float)) and not isinstance(
                        retrieval_score, bool
                    ):
                        retrieval_scores.append(float(retrieval_score))
                    else:
                        # Compatibility with older raw RAG results.
                        score = result.get("score")
                        if isinstance(score, (int, float)) and not isinstance(score, bool):
                            retrieval_scores.append(float(score))

                    reranker_score = result.get("reranker_score")
                    if isinstance(reranker_score, (int, float)) and not isinstance(
                        reranker_score, bool
                    ):
                        reranker_scores.append(float(reranker_score))

    rag_unique_chunks_count = len(rag_chunk_ids)

    retrieval_score_min = min(retrieval_scores) if retrieval_scores else None
    retrieval_score_max = max(retrieval_scores) if retrieval_scores else None
    retrieval_score_avg = (
        sum(retrieval_scores) / len(retrieval_scores) if retrieval_scores else None
    )

    reranker_score_min = min(reranker_scores) if reranker_scores else None
    reranker_score_max = max(reranker_scores) if reranker_scores else None
    reranker_score_avg = sum(reranker_scores) / len(reranker_scores) if reranker_scores else None

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

    (
        context_assembly_events_total,
        context_messages_total,
        context_estimated_tokens_total,
        context_estimated_tokens_max,
        context_budget_exceeded,
        context_source_profile_changes,
        context_source_counts,
    ) = _extract_context_diagnostics(events)

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
        retrieval_score_min=retrieval_score_min,
        retrieval_score_max=retrieval_score_max,
        retrieval_score_avg=retrieval_score_avg,
        reranker_score_min=reranker_score_min,
        reranker_score_max=reranker_score_max,
        reranker_score_avg=reranker_score_avg,
        effective_model=effective_model,
        effective_provider=effective_provider,
        model_policy_id=model_policy_id,
        model_policy_version=model_policy_version,
        context_assembly_events_total=context_assembly_events_total,
        context_messages_total=context_messages_total,
        context_estimated_tokens_total=context_estimated_tokens_total,
        context_estimated_tokens_max=context_estimated_tokens_max,
        context_budget_exceeded=context_budget_exceeded,
        context_source_profile_changes=context_source_profile_changes,
        context_source_counts=context_source_counts,
        error_type=run.error_type,
        error_message=run.error_message,
    )
