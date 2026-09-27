from datetime import UTC, datetime, timedelta

from ai_platform.agents.evaluation.evidence import extract_evidence
from ai_platform.agents.observability import (
    AgentExecutionEvent,
    AgentExecutionEventType,
)
from app.control_plane.agent_run_steps.models import (
    AgentRunStep,
    AgentRunStepStatus,
)
from app.control_plane.agent_runs.models import AgentRun, AgentRunStatus


def _run(
    *,
    status: AgentRunStatus = AgentRunStatus.COMPLETED,
) -> AgentRun:
    started_at = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)

    return AgentRun(
        run_id="run-1",
        agent_name="test-agent",
        tenant_id="tenant-1",
        status=status,
        started_at=started_at,
        completed_at=started_at + timedelta(milliseconds=1250),
        metadata={"agent_version": "1.2.3"},
    )


def _step(
    *,
    step_id: str,
    step_index: int,
    status: AgentRunStepStatus,
    tool_name: str | None = None,
    call_id: str | None = None,
    failure_category: str | None = None,
    input: object | None = None,
    output: object | None = None,
    metadata: dict[str, object] | None = None,
) -> AgentRunStep:
    return AgentRunStep(
        run_id="run-1",
        step_id=step_id,
        step_index=step_index,
        step_type="tool" if tool_name or call_id else "orchestration",
        status=status,
        tool_name=tool_name,
        call_id=call_id,
        failure_category=failure_category,
        input=input,
        output=output,
        metadata=metadata or {},
    )


def _governance_event(
    *,
    decision: str,
    domain: str = "tool",
) -> AgentExecutionEvent:
    return AgentExecutionEvent(
        event_type=AgentExecutionEventType.GOVERNANCE_DECISION,
        agent_name="test-agent",
        run_id="run-1",
        metadata={
            "governance_domain": domain,
            "decision": decision,
        },
    )


def test_extract_evidence_counts_tool_steps_and_execution_time() -> None:
    steps = [
        _step(
            step_id="step-1",
            step_index=0,
            status=AgentRunStepStatus.COMPLETED,
            tool_name="search",
            call_id="call-1",
        ),
        _step(
            step_id="step-2",
            step_index=1,
            status=AgentRunStepStatus.FAILED,
            tool_name="search",
            call_id="call-2",
            failure_category="execution_error",
        ),
        _step(
            step_id="step-3",
            step_index=2,
            status=AgentRunStepStatus.COMPLETED,
        ),
    ]

    evidence = extract_evidence(
        _run(),
        steps,
        [],
    )

    assert evidence.execution_time_ms == 1250.0
    assert evidence.total_steps == 3
    assert evidence.tool_calls_total == 2
    assert evidence.tool_calls_successful == 1
    assert evidence.tool_calls_failed == 1
    assert evidence.invalid_tool_calls == 0
    assert evidence.governance_denials == 0
    assert evidence.agent_version == "1.2.3"


def test_extract_evidence_classifies_schema_failures_as_invalid_tool_calls() -> None:
    steps = [
        _step(
            step_id="step-1",
            step_index=0,
            status=AgentRunStepStatus.FAILED,
            tool_name="search",
            call_id="call-1",
            failure_category="schema_validation",
        ),
        _step(
            step_id="step-2",
            step_index=1,
            status=AgentRunStepStatus.FAILED,
            tool_name="search",
            call_id="call-2",
            failure_category="invalid_schema",
        ),
        _step(
            step_id="step-3",
            step_index=2,
            status=AgentRunStepStatus.FAILED,
            tool_name="search",
            call_id="call-3",
            failure_category="execution_error",
        ),
    ]

    evidence = extract_evidence(
        _run(),
        steps,
        [],
    )

    assert evidence.tool_calls_total == 3
    assert evidence.tool_calls_successful == 0
    assert evidence.invalid_tool_calls == 2
    assert evidence.tool_calls_failed == 1


def test_extract_evidence_counts_call_id_only_as_tool_step() -> None:
    steps = [
        _step(
            step_id="step-1",
            step_index=0,
            status=AgentRunStepStatus.COMPLETED,
            call_id="call-1",
        ),
    ]

    evidence = extract_evidence(
        _run(),
        steps,
        [],
    )

    assert evidence.total_steps == 1
    assert evidence.tool_calls_total == 1
    assert evidence.tool_calls_successful == 1


def test_extract_evidence_counts_governance_denials_only() -> None:
    events = [
        _governance_event(decision="deny"),
        _governance_event(decision="allow"),
        _governance_event(decision="deny", domain="model"),
        AgentExecutionEvent(
            event_type=AgentExecutionEventType.TOOL_AUTHORIZATION_DECISION,
            agent_name="test-agent",
            run_id="run-1",
            metadata={"decision": "deny"},
        ),
    ]

    evidence = extract_evidence(
        _run(),
        [],
        events,
    )

    assert evidence.governance_denials == 2


def test_extract_evidence_uses_run_status_for_task_completion() -> None:
    completed = extract_evidence(
        _run(status=AgentRunStatus.COMPLETED),
        [],
        [],
    )

    failed = extract_evidence(
        _run(status=AgentRunStatus.FAILED),
        [],
        [],
    )

    assert completed.status == "completed"
    assert failed.status == "failed"


def test_extract_evidence_preserves_run_errors_without_persisting_them_here() -> None:
    run = _run(status=AgentRunStatus.FAILED).model_copy(
        update={
            "error_type": "AgentExecutionError",
            "error_message": "execution failed",
        }
    )

    evidence = extract_evidence(run, [], [])

    assert evidence.error_type == "AgentExecutionError"
    assert evidence.error_message == "execution failed"


def test_extract_evidence_does_not_use_error_text_for_invalid_tool_classification() -> None:
    steps = [
        _step(
            step_id="step-1",
            step_index=0,
            status=AgentRunStepStatus.FAILED,
            tool_name="search",
            call_id="call-1",
            failure_category="execution_error",
        )
    ]

    steps[0] = steps[0].model_copy(
        update={"error": "schema validation failed"},
    )

    evidence = extract_evidence(
        _run(),
        steps,
        [],
    )

    assert evidence.invalid_tool_calls == 0
    assert evidence.tool_calls_failed == 1


def test_extract_evidence_handles_missing_run_timestamps() -> None:
    run = _run().model_copy(
        update={
            "started_at": None,
            "completed_at": None,
        }
    )

    evidence = extract_evidence(run, [], [])

    assert evidence.execution_time_ms == 0.0


def test_extract_evidence_counts_rag_queries_and_sources_from_provenance() -> None:
    steps = [
        _step(
            step_id="step-1",
            step_index=0,
            status=AgentRunStepStatus.COMPLETED,
            tool_name="rag.search",
            call_id="call-1",
            metadata={
                "rag_provenance": {
                    "retrieved_count": 3,
                    "sources": [
                        {"chunk_id": "chunk-1"},
                        {"chunk_id": "chunk-2"},
                        {"chunk_id": "chunk-3"},
                    ],
                }
            },
        ),
        _step(
            step_id="step-2",
            step_index=1,
            status=AgentRunStepStatus.COMPLETED,
            tool_name="rag.search",
            call_id="call-2",
            metadata={
                "rag_provenance": {
                    "sources": [
                        {"chunk_id": "chunk-4"},
                        {"chunk_id": "chunk-5"},
                    ],
                }
            },
        ),
    ]

    evidence = extract_evidence(_run(), steps, [])

    assert evidence.rag_queries_total == 2
    assert evidence.rag_sources_retrieved_total == 5


def test_extract_evidence_falls_back_to_rag_output_results() -> None:
    steps = [
        _step(
            step_id="step-1",
            step_index=0,
            status=AgentRunStepStatus.COMPLETED,
            tool_name="rag.search",
            call_id="call-1",
            output={
                "query": "What is RAG?",
                "results": [
                    {"chunk_id": "chunk-1"},
                    {"chunk_id": "chunk-2"},
                ],
                "retrieved_count": 2,
            },
        )
    ]

    evidence = extract_evidence(_run(), steps, [])

    assert evidence.rag_queries_total == 1
    assert evidence.rag_sources_retrieved_total == 2


def test_extract_evidence_defaults_rag_metrics_for_non_rag_runs() -> None:
    steps = [
        _step(
            step_id="step-1",
            step_index=0,
            status=AgentRunStepStatus.COMPLETED,
            tool_name="search",
            call_id="call-1",
        )
    ]

    evidence = extract_evidence(_run(), steps, [])

    assert evidence.rag_queries_total == 0
    assert evidence.rag_sources_retrieved_total == 0
