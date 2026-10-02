from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from ai_platform.agents.observability import (
    AgentExecutionEvent,
    AgentExecutionEventType,
)
from app.control_plane.agent_run_steps.models import (
    AgentRunStep,
    AgentRunStepStatus,
)
from app.control_plane.agent_run_timeline.models import (
    AgentRunTimeline,
    AgentRunTimelineEntryKind,
    AttemptSummary,
    TimelineEntry,
)

_STEP_STARTED = AgentExecutionEventType.ORCHESTRATION_STEP_STARTED
_STEP_COMPLETED = AgentExecutionEventType.ORCHESTRATION_STEP_COMPLETED
_STEP_FAILED = AgentExecutionEventType.ORCHESTRATION_STEP_FAILED
_STEP_CANCELLED = AgentExecutionEventType.ORCHESTRATION_STEP_CANCELLED

_STEP_TERMINAL_STATUS_BY_EVENT: dict[
    AgentExecutionEventType,
    AgentRunStepStatus,
] = {
    _STEP_COMPLETED: AgentRunStepStatus.COMPLETED,
    _STEP_FAILED: AgentRunStepStatus.FAILED,
    _STEP_CANCELLED: AgentRunStepStatus.CANCELLED,
}


@dataclass
class _AttemptAccumulator:
    step_id: str
    attempt: int
    step_index: int | None
    step_name: str | None
    status: AgentRunStepStatus
    event_count: int
    first_sequence: int
    last_sequence: int
    failure_category: str | None = None


def reconstruct_timeline(
    events: Iterable[AgentExecutionEvent],
    step_snapshots: Iterable[AgentRunStep] | None = None,
) -> AgentRunTimeline:
    """
    Reconstruct a deterministic execution timeline from historical events.

    The event iterable is expected to already be ordered chronologically by
    the persistence layer. No timestamp ordering is attempted because
    AgentExecutionEvent deliberately does not contain persistence timestamps.

    Historical attempt lineage comes from orchestration step events carrying
    their explicit attempt number. Current AgentRunStep snapshots are used
    only to enrich missing step metadata and never to create historical
    attempts.
    """
    event_list = list(events)
    snapshot_list = list(step_snapshots or ())

    run_id = _resolve_run_id(event_list, snapshot_list)

    snapshot_by_step_id = {snapshot.step_id: snapshot for snapshot in snapshot_list}

    entries: list[TimelineEntry] = []
    attempts: dict[tuple[str, int], _AttemptAccumulator] = {}

    for sequence, event in enumerate(event_list):
        snapshot = snapshot_by_step_id.get(event.step_id) if event.step_id is not None else None

        step_id = event.step_id
        step_index = (
            event.step_index
            if event.step_index is not None
            else snapshot.step_index if snapshot is not None else None
        )
        step_name = (
            event.step_name if event.step_name is not None else _snapshot_step_name(snapshot)
        )

        entries.append(
            TimelineEntry(
                sequence=sequence,
                kind=(
                    AgentRunTimelineEntryKind.STEP_ATTEMPT
                    if _is_step_event(event.event_type)
                    else AgentRunTimelineEntryKind.EVENT
                ),
                event_type=event.event_type,
                run_id=event.run_id,
                agent_name=event.agent_name,
                step_id=step_id,
                step_index=step_index,
                step_name=step_name,
                attempt=event.attempt,
                tool_name=event.tool_name,
                call_id=event.call_id,
                provider=event.provider,
                model=event.model,
                metadata=dict(event.metadata),
            )
        )

        if step_id is None or event.attempt is None or not _is_step_event(event.event_type):
            continue

        key = (step_id, event.attempt)

        accumulator = attempts.get(key)
        if accumulator is None:
            accumulator = _AttemptAccumulator(
                step_id=step_id,
                attempt=event.attempt,
                step_index=step_index,
                step_name=step_name,
                status=AgentRunStepStatus.RUNNING,
                event_count=0,
                first_sequence=sequence,
                last_sequence=sequence,
            )
            attempts[key] = accumulator

        accumulator.event_count += 1
        accumulator.last_sequence = sequence

        if accumulator.step_index is None:
            accumulator.step_index = step_index

        if accumulator.step_name is None:
            accumulator.step_name = step_name

        terminal_status = _STEP_TERMINAL_STATUS_BY_EVENT.get(event.event_type)
        if terminal_status is not None:
            accumulator.status = terminal_status

        failure_category = event.metadata.get("failure_category")
        if isinstance(failure_category, str) and failure_category.strip():
            accumulator.failure_category = failure_category

    attempt_summaries = tuple(
        AttemptSummary(
            step_id=accumulator.step_id,
            attempt=accumulator.attempt,
            step_index=accumulator.step_index,
            step_name=accumulator.step_name,
            status=accumulator.status,
            event_count=accumulator.event_count,
            first_sequence=accumulator.first_sequence,
            last_sequence=accumulator.last_sequence,
            failure_category=accumulator.failure_category,
        )
        for accumulator in sorted(
            attempts.values(),
            key=lambda item: item.first_sequence,
        )
    )

    return AgentRunTimeline(
        run_id=run_id,
        entries=tuple(entries),
        attempts=attempt_summaries,
    )


def _resolve_run_id(
    events: list[AgentExecutionEvent],
    step_snapshots: Iterable[AgentRunStep] | None,
) -> str:
    event_run_ids = {event.run_id for event in events if event.run_id is not None}

    if len(event_run_ids) == 1:
        return next(iter(event_run_ids))

    if len(event_run_ids) > 1:
        raise ValueError("Agent run timeline events contain multiple run_ids.")

    snapshot_run_ids = {snapshot.run_id for snapshot in (step_snapshots or ())}

    if len(snapshot_run_ids) == 1:
        return next(iter(snapshot_run_ids))

    if len(snapshot_run_ids) > 1:
        raise ValueError("Agent run timeline step snapshots contain multiple run_ids.")

    raise ValueError(
        "Agent run timeline requires at least one event or step snapshot " "with a run_id."
    )


def _is_step_event(event_type: AgentExecutionEventType) -> bool:
    return event_type in {
        _STEP_STARTED,
        _STEP_COMPLETED,
        _STEP_FAILED,
        _STEP_CANCELLED,
    }


def _snapshot_step_name(snapshot: AgentRunStep | None) -> str | None:
    if snapshot is None:
        return None

    metadata_name = snapshot.metadata.get("step_name")
    if isinstance(metadata_name, str) and metadata_name.strip():
        return metadata_name

    return None
