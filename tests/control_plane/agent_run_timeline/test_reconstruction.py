from ai_platform.agents.observability import (
    AgentExecutionEvent,
    AgentExecutionEventType,
)
from app.control_plane.agent_run_steps.models import AgentRunStepStatus
from app.control_plane.agent_run_timeline.models import (
    AgentRunTimelineEntryKind,
)
from app.control_plane.agent_run_timeline.reconstruction import (
    reconstruct_timeline,
)

RUN_ID = "run-1"


def _event(
    event_type: AgentExecutionEventType,
    *,
    step_id: str | None = None,
    step_index: int | None = None,
    step_name: str | None = None,
    attempt: int | None = None,
    metadata: dict[str, object] | None = None,
) -> AgentExecutionEvent:
    return AgentExecutionEvent(
        event_type=event_type,
        agent_name="test-agent",
        run_id=RUN_ID,
        step_id=step_id,
        step_index=step_index,
        step_name=step_name,
        attempt=attempt,
        metadata={} if metadata is None else metadata,
    )


def test_reconstructs_successful_step_attempt() -> None:
    timeline = reconstruct_timeline(
        [
            _event(
                AgentExecutionEventType.AGENT_STARTED,
            ),
            _event(
                AgentExecutionEventType.ORCHESTRATION_STEP_STARTED,
                step_id="step-1",
                step_index=0,
                step_name="retrieve",
                attempt=1,
            ),
            _event(
                AgentExecutionEventType.ORCHESTRATION_STEP_COMPLETED,
                step_id="step-1",
                step_index=0,
                step_name="retrieve",
                attempt=1,
            ),
            _event(
                AgentExecutionEventType.AGENT_COMPLETED,
            ),
        ]
    )

    assert timeline.run_id == RUN_ID
    assert [entry.sequence for entry in timeline.entries] == [0, 1, 2, 3]

    assert timeline.entries[1].kind is AgentRunTimelineEntryKind.STEP_ATTEMPT
    assert timeline.entries[1].attempt == 1

    assert len(timeline.attempts) == 1

    attempt = timeline.attempts[0]

    assert attempt.step_id == "step-1"
    assert attempt.attempt == 1
    assert attempt.status is AgentRunStepStatus.COMPLETED
    assert attempt.event_count == 2
    assert attempt.first_sequence == 1
    assert attempt.last_sequence == 2


def test_reconstructs_failed_attempt_followed_by_successful_retry() -> None:
    timeline = reconstruct_timeline(
        [
            _event(
                AgentExecutionEventType.ORCHESTRATION_STEP_STARTED,
                step_id="step-2",
                step_index=1,
                attempt=1,
            ),
            _event(
                AgentExecutionEventType.ORCHESTRATION_STEP_FAILED,
                step_id="step-2",
                step_index=1,
                attempt=1,
                metadata={
                    "failure_category": "provider_timeout",
                },
            ),
            _event(
                AgentExecutionEventType.ORCHESTRATION_STEP_STARTED,
                step_id="step-2",
                step_index=1,
                attempt=2,
            ),
            _event(
                AgentExecutionEventType.ORCHESTRATION_STEP_COMPLETED,
                step_id="step-2",
                step_index=1,
                attempt=2,
            ),
        ]
    )

    assert len(timeline.attempts) == 2

    first, second = timeline.attempts

    assert first.step_id == "step-2"
    assert first.attempt == 1
    assert first.status is AgentRunStepStatus.FAILED
    assert first.failure_category == "provider_timeout"
    assert first.first_sequence == 0
    assert first.last_sequence == 1

    assert second.step_id == "step-2"
    assert second.attempt == 2
    assert second.status is AgentRunStepStatus.COMPLETED
    assert second.first_sequence == 2
    assert second.last_sequence == 3


def test_page_local_reconstruction_handles_attempt_split_across_pages() -> None:
    page_one = reconstruct_timeline(
        [
            _event(
                AgentExecutionEventType.ORCHESTRATION_STEP_STARTED,
                step_id="step-2",
                step_index=1,
                attempt=1,
            ),
        ]
    )

    page_two = reconstruct_timeline(
        [
            _event(
                AgentExecutionEventType.ORCHESTRATION_STEP_FAILED,
                step_id="step-2",
                step_index=1,
                attempt=1,
                metadata={
                    "failure_category": "provider_timeout",
                },
            ),
            _event(
                AgentExecutionEventType.ORCHESTRATION_STEP_STARTED,
                step_id="step-2",
                step_index=1,
                attempt=2,
            ),
            _event(
                AgentExecutionEventType.ORCHESTRATION_STEP_COMPLETED,
                step_id="step-2",
                step_index=1,
                attempt=2,
            ),
        ]
    )

    assert page_one.run_id == RUN_ID
    assert len(page_one.entries) == 1
    assert page_one.entries[0].sequence == 0

    assert len(page_one.attempts) == 1
    assert page_one.attempts[0].attempt == 1
    assert page_one.attempts[0].status is AgentRunStepStatus.RUNNING
    assert page_one.attempts[0].event_count == 1
    assert page_one.attempts[0].first_sequence == 0
    assert page_one.attempts[0].last_sequence == 0

    assert page_two.run_id == RUN_ID
    assert [entry.sequence for entry in page_two.entries] == [0, 1, 2]

    assert [
        (attempt.step_id, attempt.attempt, attempt.status, attempt.event_count)
        for attempt in page_two.attempts
    ] == [
        ("step-2", 1, AgentRunStepStatus.FAILED, 1),
        ("step-2", 2, AgentRunStepStatus.COMPLETED, 2),
    ]

    assert page_two.attempts[0].failure_category == "provider_timeout"
    assert page_two.attempts[0].first_sequence == 0
    assert page_two.attempts[0].last_sequence == 0

    assert page_two.attempts[1].first_sequence == 1
    assert page_two.attempts[1].last_sequence == 2


def test_preserves_interleaved_events_and_attempt_order() -> None:
    timeline = reconstruct_timeline(
        [
            _event(
                AgentExecutionEventType.ORCHESTRATION_STEP_STARTED,
                step_id="step-1",
                step_index=0,
                attempt=1,
            ),
            _event(
                AgentExecutionEventType.ORCHESTRATION_STEP_STARTED,
                step_id="step-2",
                step_index=1,
                attempt=1,
            ),
            _event(
                AgentExecutionEventType.ORCHESTRATION_STEP_COMPLETED,
                step_id="step-2",
                step_index=1,
                attempt=1,
            ),
            _event(
                AgentExecutionEventType.ORCHESTRATION_STEP_COMPLETED,
                step_id="step-1",
                step_index=0,
                attempt=1,
            ),
        ]
    )

    assert [(entry.sequence, entry.step_id, entry.attempt) for entry in timeline.entries] == [
        (0, "step-1", 1),
        (1, "step-2", 1),
        (2, "step-2", 1),
        (3, "step-1", 1),
    ]

    assert [
        (attempt.step_id, attempt.first_sequence, attempt.last_sequence)
        for attempt in timeline.attempts
    ] == [
        ("step-1", 0, 3),
        ("step-2", 1, 2),
    ]


def test_incomplete_attempt_remains_running() -> None:
    timeline = reconstruct_timeline(
        [
            _event(
                AgentExecutionEventType.ORCHESTRATION_STEP_STARTED,
                step_id="step-3",
                step_index=2,
                attempt=1,
            ),
        ]
    )

    assert len(timeline.attempts) == 1
    assert timeline.attempts[0].status is AgentRunStepStatus.RUNNING


def test_cancelled_attempt_is_reconstructed() -> None:
    timeline = reconstruct_timeline(
        [
            _event(
                AgentExecutionEventType.ORCHESTRATION_STEP_STARTED,
                step_id="step-4",
                step_index=3,
                attempt=1,
            ),
            _event(
                AgentExecutionEventType.ORCHESTRATION_STEP_CANCELLED,
                step_id="step-4",
                step_index=3,
                attempt=1,
            ),
        ]
    )

    assert timeline.attempts[0].status is AgentRunStepStatus.CANCELLED


def test_non_step_events_are_preserved_without_creating_attempts() -> None:
    timeline = reconstruct_timeline(
        [
            _event(
                AgentExecutionEventType.AGENT_STARTED,
            ),
            _event(
                AgentExecutionEventType.LLM_REQUESTED,
            ),
            _event(
                AgentExecutionEventType.LLM_COMPLETED,
            ),
            _event(
                AgentExecutionEventType.AGENT_COMPLETED,
            ),
        ]
    )

    assert len(timeline.entries) == 4
    assert all(entry.kind is AgentRunTimelineEntryKind.EVENT for entry in timeline.entries)
    assert timeline.attempts == ()


def test_snapshot_enriches_missing_step_metadata_without_creating_attempt() -> None:
    from app.control_plane.agent_run_steps.models import AgentRunStep

    snapshot = AgentRunStep(
        run_id=RUN_ID,
        step_id="step-5",
        step_index=4,
        step_type="tool",
        status=AgentRunStepStatus.COMPLETED,
        attempt=2,
        metadata={"step_name": "query_vehicle"},
    )

    timeline = reconstruct_timeline(
        [
            _event(
                AgentExecutionEventType.ORCHESTRATION_STEP_STARTED,
                step_id="step-5",
                attempt=2,
            ),
            _event(
                AgentExecutionEventType.ORCHESTRATION_STEP_COMPLETED,
                step_id="step-5",
                attempt=2,
            ),
        ],
        step_snapshots=[snapshot],
    )

    assert timeline.entries[0].step_index == 4
    assert timeline.entries[0].step_name == "query_vehicle"
    assert timeline.attempts[0].step_index == 4
    assert timeline.attempts[0].step_name == "query_vehicle"


def test_accepts_one_shot_step_snapshot_iterable() -> None:
    from app.control_plane.agent_run_steps.models import AgentRunStep

    snapshot = AgentRunStep(
        run_id=RUN_ID,
        step_id="step-generator",
        step_index=0,
        step_type="tool",
        status=AgentRunStepStatus.COMPLETED,
        attempt=1,
        metadata={"step_name": "generator_step"},
    )

    snapshots = iter([snapshot])

    timeline = reconstruct_timeline([], snapshots)

    assert timeline.run_id == RUN_ID
    assert timeline.entries == ()
    assert timeline.attempts == ()


def test_multiple_run_ids_are_rejected() -> None:
    first = _event(AgentExecutionEventType.AGENT_STARTED)

    second = AgentExecutionEvent(
        event_type=AgentExecutionEventType.AGENT_COMPLETED,
        agent_name="test-agent",
        run_id="run-2",
    )

    try:
        reconstruct_timeline([first, second])
    except ValueError as exc:
        assert "multiple run_ids" in str(exc)
    else:
        raise AssertionError("Expected ValueError")
