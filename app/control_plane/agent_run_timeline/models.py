from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from app.control_plane.agent_run_steps.models import AgentRunStepStatus
from ai_platform.agents.observability import AgentExecutionEventType


class AgentRunTimelineEntryKind(StrEnum):
    EVENT = "event"
    STEP_ATTEMPT = "step_attempt"


@dataclass(frozen=True)
class TimelineEntry:
    """
    One chronological execution-history entry.

    Sequence is assigned from the order supplied by the event repository.
    AgentExecutionEvent intentionally does not expose persistence timestamps,
    so reconstruction must not invent timestamps.
    """

    sequence: int
    kind: AgentRunTimelineEntryKind
    event_type: AgentExecutionEventType
    run_id: str | None
    agent_name: str
    step_id: str | None
    step_index: int | None
    step_name: str | None
    attempt: int | None
    tool_name: str | None
    call_id: str | None
    provider: str | None
    model: str | None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AttemptSummary:
    """
    Reconstructed historical state for one step attempt.

    This is derived exclusively from the event stream. It must not be
    interpreted as a persisted attempt record.
    """

    step_id: str
    attempt: int
    step_index: int | None
    step_name: str | None
    status: AgentRunStepStatus
    event_count: int
    first_sequence: int
    last_sequence: int
    failure_category: str | None = None


@dataclass(frozen=True)
class AgentRunTimeline:
    """
    Deterministic execution-history projection for one agent run.
    """

    run_id: str
    entries: tuple[TimelineEntry, ...]
    attempts: tuple[AttemptSummary, ...]


@dataclass(frozen=True)
class AgentRunTimelinePage:
    """
    A reconstructed timeline for one historical event page.

    Pagination metadata is inherited from the event repository. The
    reconstructed timeline therefore represents only the events returned
    for this page, not an implicit complete-history projection.
    """

    timeline: AgentRunTimeline
    next_cursor: str | None
    has_more: bool
