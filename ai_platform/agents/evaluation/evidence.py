from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
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


def _canonicalize_evidence_value(value: Any) -> Any:
    """Return a deterministic JSON-compatible representation of evidence."""
    if isinstance(value, datetime):
        normalized = (
            value.astimezone(UTC) if value.tzinfo is not None else value.replace(tzinfo=UTC)
        )
        return normalized.isoformat()

    if isinstance(value, dict):
        return {
            str(key): _canonicalize_evidence_value(value[key])
            for key in sorted(value, key=lambda item: str(item))
        }

    if isinstance(value, (list, tuple)):
        return [_canonicalize_evidence_value(item) for item in value]

    if isinstance(value, set | frozenset):
        canonical_items = [_canonicalize_evidence_value(item) for item in value]
        return sorted(
            canonical_items,
            key=lambda item: json.dumps(
                item,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
        )

    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError("evidence fingerprint cannot encode non-finite floats")

    return value


def _canonical_rag_source(source: RagEvidenceSource) -> dict[str, Any]:
    return {
        "source_index": source.source_index,
        "content": source.content,
        "chunk_id": source.chunk_id,
        "document_id": source.document_id,
        "retrieval_score": source.retrieval_score,
        "reranker_score": source.reranker_score,
    }


def _canonical_evidence_input(
    run: AgentRun,
    steps: list[AgentRunStep],
    events: list[AgentExecutionEvent],
) -> dict[str, Any]:
    """Build a deterministic projection of durable evaluation source evidence."""

    def step_payload(step: AgentRunStep) -> dict[str, Any]:
        return {
            "run_id": step.run_id,
            "step_id": step.step_id,
            "step_index": step.step_index,
            "step_type": step.step_type,
            "status": step.status.value,
            "attempt": step.attempt,
            "tool_name": step.tool_name,
            "call_id": step.call_id,
            "input": step.input,
            "output": step.output,
            "error": step.error,
            "failure_category": step.failure_category,
            "started_at": step.started_at,
            "completed_at": step.completed_at,
            "metadata": step.metadata,
        }

    def event_payload(event: AgentExecutionEvent) -> dict[str, Any]:
        return {
            "event_type": event.event_type.value,
            "agent_name": event.agent_name,
            "run_id": event.run_id,
            "session_id": event.session_id,
            "user_id": event.user_id,
            "principal": event.principal,
            "tool_round": event.tool_round,
            "tool_name": event.tool_name,
            "call_id": event.call_id,
            "provider": event.provider,
            "model": event.model,
            "step_id": event.step_id,
            "step_index": event.step_index,
            "step_name": event.step_name,
            "attempt": event.attempt,
            "metadata": event.metadata,
        }

    def canonical_sort_key(payload: dict[str, Any]) -> tuple[Any, ...]:
        return (
            payload.get("step_index") if payload.get("step_index") is not None else -1,
            payload.get("step_id") or "",
            payload.get("attempt") if payload.get("attempt") is not None else 0,
            payload.get("tool_round") if payload.get("tool_round") is not None else -1,
            payload.get("call_id") or "",
            payload.get("event_type") or "",
            payload.get("provider") or "",
            payload.get("model") or "",
            payload.get("agent_name") or "",
            json.dumps(
                _canonicalize_evidence_value(payload),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
        )

    canonical_step_payloads = [step_payload(step) for step in steps]
    canonical_steps = sorted(
        canonical_step_payloads,
        key=canonical_sort_key,
    )

    canonical_event_payloads = [event_payload(event) for event in events]
    canonical_events = sorted(
        canonical_event_payloads,
        key=canonical_sort_key,
    )

    canonical_step_models = sorted(
        steps,
        key=lambda step: (
            step.step_index,
            step.step_id,
            step.attempt,
            json.dumps(
                _canonicalize_evidence_value(step_payload(step)),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
        ),
    )

    rag_sources = extract_rag_evidence_sources(canonical_step_models)

    return _canonicalize_evidence_value(
        {
            "run": {
                "run_id": run.run_id,
                "agent_name": run.agent_name,
                "tenant_id": run.tenant_id,
                "status": run.status.value,
                "started_at": run.started_at,
                "completed_at": run.completed_at,
                "output": run.output,
                "metadata": run.metadata,
                "request_snapshot": (
                    run.request_snapshot.model_dump(mode="json")
                    if run.request_snapshot is not None
                    else None
                ),
                "error_type": run.error_type,
                "error_message": run.error_message,
            },
            "steps": canonical_steps,
            "events": canonical_events,
            "rag_sources": [_canonical_rag_source(source) for source in rag_sources],
        }
    )


def compute_evidence_fingerprint(
    run: AgentRun,
    steps: list[AgentRunStep],
    events: list[AgentExecutionEvent],
) -> str:
    """Return a deterministic SHA-256 fingerprint of evaluation source evidence."""
    canonical_input = _canonical_evidence_input(run, steps, events)
    payload = json.dumps(
        canonical_input,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    return hashlib.sha256(payload).hexdigest()


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
) -> tuple[
    int,
    int,
    int,
    int,
    int | None,
    bool,
    int,
    dict[str, int],
    tuple[tuple[dict[str, object], ...], ...],
]:
    """Aggregate bounded context-assembly diagnostics from durable events."""
    context_assembly_events_total = 0
    context_messages_total = 0
    context_estimated_tokens_total = 0
    context_estimated_tokens_max = 0
    context_estimated_remaining_after_context_min: int | None = None
    context_budget_exceeded = False
    context_source_profile_changes = 0
    context_source_counts: dict[str, int] = {}
    context_source_lineage: list[tuple[dict[str, object], ...]] = []
    previous_source_profile: frozenset[str] | None = None

    for event in events:
        if event.event_type != AgentExecutionEventType.CONTEXT_ASSEMBLY_COMPLETED:
            continue

        context_assembly_events_total += 1
        metadata = event.metadata

        source_lineage = metadata.get("source_lineage")
        if isinstance(source_lineage, (list, tuple)):
            assembly_lineage: list[dict[str, object]] = []

            for source in source_lineage:
                if not isinstance(source, dict):
                    continue

                entry = dict(source)

                provenance = entry.get("provenance")
                if isinstance(provenance, dict):
                    entry["provenance"] = dict(provenance)

                assembly_lineage.append(entry)

            context_source_lineage.append(tuple(assembly_lineage))

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

            remaining_after_context = budget_status.get("estimated_remaining_after_context")
            if isinstance(remaining_after_context, int) and not isinstance(
                remaining_after_context, bool
            ):
                if (
                    context_estimated_remaining_after_context_min is None
                    or remaining_after_context < context_estimated_remaining_after_context_min
                ):
                    context_estimated_remaining_after_context_min = remaining_after_context

    return (
        context_assembly_events_total,
        context_messages_total,
        context_estimated_tokens_total,
        context_estimated_tokens_max,
        context_estimated_remaining_after_context_min,
        context_budget_exceeded,
        context_source_profile_changes,
        context_source_counts,
        tuple(context_source_lineage),
    )


@dataclass(frozen=True)
class RagEvidenceSource:
    """Transient RAG source evidence used by evaluation.

    Raw source content remains in-memory only. Identity and retrieval metadata
    are retained so downstream evaluation can attribute answer claims to the
    exact retrieved source without expanding the persisted evaluation schema.
    """

    source_index: int
    content: str
    chunk_id: str | None = None
    document_id: str | None = None
    retrieval_score: float | None = None
    reranker_score: float | None = None


def _extract_rag_source(
    result: dict[str, Any],
    source_index: int,
) -> RagEvidenceSource | None:
    content = result.get("content")
    if not isinstance(content, str) or not content.strip():
        return None

    chunk_id = result.get("chunk_id")
    if not isinstance(chunk_id, str) or not chunk_id:
        chunk_id = None

    document_id = result.get("document_id")
    if not isinstance(document_id, str) or not document_id:
        document_id = None

    retrieval_score = result.get("retrieval_score")
    if not isinstance(retrieval_score, (int, float)) or isinstance(retrieval_score, bool):
        retrieval_score = result.get("score")
    if not isinstance(retrieval_score, (int, float)) or isinstance(retrieval_score, bool):
        retrieval_score = None
    else:
        retrieval_score = float(retrieval_score)

    reranker_score = result.get("reranker_score")
    if not isinstance(reranker_score, (int, float)) or isinstance(reranker_score, bool):
        reranker_score = None
    else:
        reranker_score = float(reranker_score)

    return RagEvidenceSource(
        source_index=source_index,
        content=content.strip(),
        chunk_id=chunk_id,
        document_id=document_id,
        retrieval_score=retrieval_score,
        reranker_score=reranker_score,
    )


def extract_rag_evidence_sources(
    steps: list[AgentRunStep],
) -> list[RagEvidenceSource]:
    """Extract structured RAG evidence transiently for evaluation.

    Source content is intentionally not persisted in AgentRunEvidence or
    evaluation metrics. The returned objects retain source identity and
    retrieval metadata so evaluation can later attribute claims to sources.
    """

    sources: list[RagEvidenceSource] = []

    for step in steps:
        provenance = step.metadata.get("rag_provenance")
        is_rag_step = step.tool_name == "rag.search" or isinstance(provenance, dict)

        if not is_rag_step:
            continue

        output = step.output
        if not isinstance(output, dict):
            continue

        results = output.get("results")
        if not isinstance(results, (list, tuple)):
            continue

        for result in results:
            if not isinstance(result, dict):
                continue

            source = _extract_rag_source(
                result,
                source_index=len(sources),
            )
            if source is not None:
                sources.append(source)

    return sources


def extract_rag_source_texts(
    steps: list[AgentRunStep],
) -> list[str]:
    """Extract raw RAG source text transiently for grounding evaluation.

    This compatibility wrapper preserves the existing text-only grounding
    contract while structured evidence is available to newer consumers.
    """

    return [source.content for source in extract_rag_evidence_sources(steps)]


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
    has_rag_provenance = False
    rag_chunk_ids: set[str] = set()

    retrieval_scores: list[float] = []
    reranker_scores: list[float] = []

    for step in steps:
        provenance = step.metadata.get("rag_provenance")
        is_rag_step = step.tool_name == "rag.search" or isinstance(provenance, dict)

        if not is_rag_step:
            continue

        rag_queries_total += 1

        if isinstance(provenance, dict) and provenance:
            has_rag_provenance = True
            retrieved_count = provenance.get("retrieved_count")
            sources = provenance.get("sources")

            if isinstance(sources, (list, tuple)):
                valid_sources = [source for source in sources if isinstance(source, dict)]

                if valid_sources:
                    if isinstance(retrieved_count, int) and retrieved_count >= 0:
                        rag_sources_retrieved_total += retrieved_count
                    else:
                        rag_sources_retrieved_total += len(valid_sources)

                    rag_sources_available_count += len(valid_sources)

                    for source in valid_sources:
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
        context_estimated_remaining_after_context_min,
        context_budget_exceeded,
        context_source_profile_changes,
        context_source_counts,
        context_source_lineage,
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
        has_rag_provenance=has_rag_provenance,
        has_rag_sources_available=rag_sources_available_count > 0,
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
        context_estimated_remaining_after_context_min=(
            context_estimated_remaining_after_context_min
        ),
        context_budget_exceeded=context_budget_exceeded,
        context_source_profile_changes=context_source_profile_changes,
        context_source_counts=context_source_counts,
        context_source_lineage=context_source_lineage,
        error_type=run.error_type,
        error_message=run.error_message,
    )
