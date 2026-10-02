from datetime import UTC, datetime, timedelta, timezone

from ai_platform.agents.evaluation.evidence import (
    extract_evidence,
    extract_rag_evidence_sources,
    extract_rag_source_texts,
)
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


def test_extract_rag_evidence_sources_preserves_identity_and_scores() -> None:
    steps = [
        _step(
            step_id="step-1",
            step_index=0,
            status=AgentRunStepStatus.COMPLETED,
            tool_name="rag.search",
            call_id="call-1",
            output={
                "results": [
                    {
                        "content": "Vehicle V001 reported 72 km/h.",
                        "chunk_id": "chunk-1",
                        "document_id": "doc-1",
                        "retrieval_score": 0.82,
                        "reranker_score": 0.94,
                    },
                    {
                        "content": "Vehicle V001 reported 68 km/h.",
                        "chunk_id": "chunk-2",
                        "document_id": "doc-1",
                        "score": 0.71,
                    },
                ]
            },
        )
    ]

    sources = extract_rag_evidence_sources(steps)

    assert len(sources) == 2

    assert sources[0].source_index == 0
    assert sources[0].content == "Vehicle V001 reported 72 km/h."
    assert sources[0].chunk_id == "chunk-1"
    assert sources[0].document_id == "doc-1"
    assert sources[0].retrieval_score == 0.82
    assert sources[0].reranker_score == 0.94

    assert sources[1].source_index == 1
    assert sources[1].content == "Vehicle V001 reported 68 km/h."
    assert sources[1].chunk_id == "chunk-2"
    assert sources[1].document_id == "doc-1"
    assert sources[1].retrieval_score == 0.71
    assert sources[1].reranker_score is None


def test_extract_rag_source_texts_remains_compatible() -> None:
    steps = [
        _step(
            step_id="step-1",
            step_index=0,
            status=AgentRunStepStatus.COMPLETED,
            tool_name="rag.search",
            call_id="call-1",
            output={
                "results": [
                    {"content": "First source.", "chunk_id": "chunk-1"},
                    {"content": "Second source.", "chunk_id": "chunk-2"},
                ]
            },
        )
    ]

    assert extract_rag_source_texts(steps) == [
        "First source.",
        "Second source.",
    ]


def test_extract_rag_evidence_sources_ignores_invalid_source_content() -> None:
    steps = [
        _step(
            step_id="step-1",
            step_index=0,
            status=AgentRunStepStatus.COMPLETED,
            tool_name="rag.search",
            call_id="call-1",
            output={
                "results": [
                    {"chunk_id": "chunk-ignored"},
                    {"content": ""},
                    {"content": "  "},
                    {"content": "Valid source.", "chunk_id": "chunk-valid"},
                ]
            },
        )
    ]

    sources = extract_rag_evidence_sources(steps)

    assert len(sources) == 1
    assert sources[0].source_index == 0
    assert sources[0].content == "Valid source."
    assert sources[0].chunk_id == "chunk-valid"


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


def test_extract_evidence_aggregates_rag_score_signals() -> None:
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
                        {
                            "chunk_id": "chunk-1",
                            "retrieval_score": 0.72,
                            "reranker_score": 0.91,
                        },
                        {
                            "chunk_id": "chunk-2",
                            "retrieval_score": 0.61,
                            "reranker_score": 0.88,
                        },
                        {
                            "chunk_id": "chunk-3",
                            "retrieval_score": 0.83,
                            "reranker_score": 0.95,
                        },
                    ],
                }
            },
        ),
    ]

    evidence = extract_evidence(_run(), steps, [])

    assert evidence.retrieval_score_min == 0.61
    assert evidence.retrieval_score_max == 0.83
    assert evidence.retrieval_score_avg == (0.72 + 0.61 + 0.83) / 3

    assert evidence.reranker_score_min == 0.88
    assert evidence.reranker_score_max == 0.95
    assert evidence.reranker_score_avg == (0.91 + 0.88 + 0.95) / 3


def test_extract_evidence_uses_legacy_score_as_retrieval_score() -> None:
    steps = [
        _step(
            step_id="step-1",
            step_index=0,
            status=AgentRunStepStatus.COMPLETED,
            tool_name="rag.search",
            call_id="call-1",
            metadata={
                "rag_provenance": {
                    "retrieved_count": 2,
                    "sources": [
                        {"chunk_id": "chunk-1", "score": 0.70},
                        {"chunk_id": "chunk-2", "score": 0.90},
                    ],
                }
            },
        ),
    ]

    evidence = extract_evidence(_run(), steps, [])

    assert evidence.retrieval_score_min == 0.70
    assert evidence.retrieval_score_max == 0.90
    assert evidence.retrieval_score_avg == 0.80
    assert evidence.reranker_score_min is None
    assert evidence.reranker_score_max is None
    assert evidence.reranker_score_avg is None


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


def test_extract_evidence_falls_back_to_raw_results_when_provenance_sources_are_missing() -> None:
    steps = [
        _step(
            step_id="step-1",
            step_index=0,
            status=AgentRunStepStatus.COMPLETED,
            tool_name="rag.search",
            call_id="call-1",
            metadata={
                "rag_provenance": {
                    "retrieved_count": 2,
                }
            },
            output={
                "results": [
                    {
                        "chunk_id": "chunk-1",
                        "retrieval_score": 0.82,
                        "reranker_score": 0.91,
                    },
                    {
                        "chunk_id": "chunk-2",
                        "retrieval_score": 0.74,
                        "reranker_score": 0.88,
                    },
                ]
            },
        )
    ]

    evidence = extract_evidence(_run(), steps, [])

    assert evidence.has_rag_provenance is True
    assert evidence.rag_sources_retrieved_total == 2
    assert evidence.rag_sources_available_count == 2
    assert evidence.rag_unique_chunks_count == 2
    assert evidence.retrieval_score_min == 0.74
    assert evidence.retrieval_score_max == 0.82
    assert evidence.retrieval_score_avg == (0.82 + 0.74) / 2
    assert evidence.reranker_score_min == 0.88
    assert evidence.reranker_score_max == 0.91
    assert evidence.reranker_score_avg == (0.91 + 0.88) / 2


def test_extract_evidence_falls_back_to_raw_results_when_provenance_sources_are_malformed() -> None:
    steps = [
        _step(
            step_id="step-1",
            step_index=0,
            status=AgentRunStepStatus.COMPLETED,
            tool_name="rag.search",
            call_id="call-1",
            metadata={
                "rag_provenance": {
                    "retrieved_count": 1,
                    "sources": "malformed",
                }
            },
            output={
                "results": [
                    {
                        "chunk_id": "chunk-1",
                        "score": 0.79,
                    }
                ]
            },
        )
    ]

    evidence = extract_evidence(_run(), steps, [])

    assert evidence.has_rag_provenance is True
    assert evidence.rag_sources_retrieved_total == 1
    assert evidence.rag_sources_available_count == 1
    assert evidence.rag_unique_chunks_count == 1
    assert evidence.retrieval_score_min == 0.79
    assert evidence.retrieval_score_max == 0.79
    assert evidence.retrieval_score_avg == 0.79


def test_extract_evidence_falls_back_to_raw_results_when_provenance_sources_are_empty() -> None:
    steps = [
        _step(
            step_id="step-1",
            step_index=0,
            status=AgentRunStepStatus.COMPLETED,
            tool_name="rag.search",
            call_id="call-1",
            metadata={
                "rag_provenance": {
                    "retrieved_count": 0,
                    "sources": [],
                }
            },
            output={
                "results": [
                    {
                        "chunk_id": "chunk-1",
                        "retrieval_score": 0.86,
                    }
                ]
            },
        )
    ]

    evidence = extract_evidence(_run(), steps, [])

    assert evidence.has_rag_provenance is True
    assert evidence.rag_sources_retrieved_total == 1
    assert evidence.rag_sources_available_count == 1
    assert evidence.rag_unique_chunks_count == 1
    assert evidence.retrieval_score_min == 0.86
    assert evidence.retrieval_score_max == 0.86
    assert evidence.retrieval_score_avg == 0.86


def test_extract_evidence_classifies_zero_source_rag_provenance() -> None:
    steps = [
        _step(
            step_id="step-1",
            step_index=0,
            status=AgentRunStepStatus.COMPLETED,
            tool_name="rag.search",
            call_id="call-1",
            metadata={
                "rag_provenance": {
                    "retrieved_count": 0,
                    "sources": [],
                }
            },
        )
    ]

    evidence = extract_evidence(_run(), steps, [])

    assert evidence.has_rag_provenance is True
    assert evidence.has_rag_sources_available is False
    assert evidence.rag_sources_available_count == 0
    assert evidence.rag_unique_chunks_count == 0
    assert evidence.rag_sources_retrieved_total == 0
    assert evidence.retrieval_score_avg is None
    assert evidence.reranker_score_avg is None


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


def test_extract_evidence_extracts_final_answer_and_unique_rag_chunks() -> None:
    run = _run().model_copy(
        update={
            "output": {
                "reply": "The vehicle exceeded the speed threshold.",
                "provider": "test-provider",
                "model": "test-model",
            }
        }
    )

    steps = [
        _step(
            step_id="step-1",
            step_index=0,
            status=AgentRunStepStatus.COMPLETED,
            tool_name="rag.search",
            metadata={
                "rag_provenance": {
                    "retrieved_count": 2,
                    "sources": [
                        {
                            "chunk_id": "chunk-1",
                            "document_id": "doc-1",
                            "score": 0.91,
                        },
                        {
                            "chunk_id": "chunk-2",
                            "document_id": "doc-1",
                            "score": 0.87,
                        },
                    ],
                }
            },
        ),
        _step(
            step_id="step-2",
            step_index=1,
            status=AgentRunStepStatus.COMPLETED,
            tool_name="rag.search",
            metadata={
                "rag_provenance": {
                    "retrieved_count": 2,
                    "sources": [
                        {
                            "chunk_id": "chunk-2",
                            "document_id": "doc-1",
                            "score": 0.88,
                        },
                        {
                            "chunk_id": "chunk-3",
                            "document_id": "doc-2",
                            "score": 0.84,
                        },
                    ],
                }
            },
        ),
    ]

    evidence = extract_evidence(run, steps, [])

    assert evidence.has_final_answer is True
    assert evidence.final_answer_length == len("The vehicle exceeded the speed threshold.")
    assert evidence.rag_sources_retrieved_total == 4
    assert evidence.rag_sources_available_count == 4
    assert evidence.rag_unique_chunks_count == 3


def test_extract_evidence_handles_plain_string_final_answer() -> None:
    run = _run().model_copy(
        update={
            "output": "  Final answer from the agent.  ",
        }
    )

    evidence = extract_evidence(run, [], [])

    assert evidence.has_final_answer is True
    assert evidence.final_answer_length == len("Final answer from the agent.")


def test_extract_evidence_handles_missing_final_answer() -> None:
    evidence = extract_evidence(_run(), [], [])

    assert evidence.has_final_answer is False
    assert evidence.final_answer_length == 0


def test_extract_evidence_counts_rag_output_results_as_available_sources() -> None:
    steps = [
        _step(
            step_id="step-1",
            step_index=0,
            status=AgentRunStepStatus.COMPLETED,
            tool_name="rag.search",
            output={
                "results": [
                    {
                        "chunk_id": "chunk-1",
                        "document_id": "doc-1",
                    },
                    {
                        "chunk_id": "chunk-2",
                        "document_id": "doc-1",
                    },
                ]
            },
        ),
    ]

    evidence = extract_evidence(_run(), steps, [])

    assert evidence.rag_sources_retrieved_total == 2
    assert evidence.rag_sources_available_count == 2
    assert evidence.rag_unique_chunks_count == 2


def test_extract_evidence_ignores_malformed_remaining_context_headroom():
    events = [
        AgentExecutionEvent(
            event_type=AgentExecutionEventType.CONTEXT_ASSEMBLY_COMPLETED,
            agent_name="vehicle-agent",
            run_id="run-1",
            metadata={
                "budget_status": {
                    "estimated_remaining_after_context": None,
                },
            },
        ),
        AgentExecutionEvent(
            event_type=AgentExecutionEventType.CONTEXT_ASSEMBLY_COMPLETED,
            agent_name="vehicle-agent",
            run_id="run-1",
            metadata={
                "budget_status": {
                    "estimated_remaining_after_context": True,
                },
            },
        ),
        AgentExecutionEvent(
            event_type=AgentExecutionEventType.CONTEXT_ASSEMBLY_COMPLETED,
            agent_name="vehicle-agent",
            run_id="run-1",
            metadata={
                "budget_status": {
                    "estimated_remaining_after_context": "200",
                },
            },
        ),
    ]

    evidence = extract_evidence(
        _run(),
        [],
        events,
    )

    assert evidence.context_estimated_remaining_after_context_min is None


def test_extract_evidence_aggregates_minimum_remaining_context_headroom():
    events = [
        AgentExecutionEvent(
            event_type=AgentExecutionEventType.CONTEXT_ASSEMBLY_COMPLETED,
            agent_name="vehicle-agent",
            run_id="run-1",
            metadata={
                "budget_status": {
                    "estimated_remaining_after_context": 500,
                },
            },
        ),
        AgentExecutionEvent(
            event_type=AgentExecutionEventType.CONTEXT_ASSEMBLY_COMPLETED,
            agent_name="vehicle-agent",
            run_id="run-1",
            metadata={
                "budget_status": {
                    "estimated_remaining_after_context": 200,
                },
            },
        ),
        AgentExecutionEvent(
            event_type=AgentExecutionEventType.CONTEXT_ASSEMBLY_COMPLETED,
            agent_name="vehicle-agent",
            run_id="run-1",
            metadata={
                "budget_status": {
                    "estimated_remaining_after_context": -50,
                },
            },
        ),
    ]

    evidence = extract_evidence(
        _run(),
        [],
        events,
    )

    assert evidence.context_estimated_remaining_after_context_min == -50


def test_extract_evidence_aggregates_context_assembly_diagnostics() -> None:
    events = [
        AgentExecutionEvent(
            event_type=AgentExecutionEventType.CONTEXT_ASSEMBLY_COMPLETED,
            agent_name="test-agent",
            run_id="run-1",
            metadata={
                "total_messages": 4,
                "estimated_tokens": 12,
                "source_counts": {
                    "system_prompt": 1,
                    "semantic_memory": 2,
                    "user_input": 1,
                },
                "budget_status": {
                    "max_tokens_per_run": 100,
                    "consumed_tokens": 20,
                    "remaining_run_tokens": 80,
                    "estimated_remaining_after_context": 68,
                    "within_budget": True,
                },
            },
        ),
        AgentExecutionEvent(
            event_type=AgentExecutionEventType.CONTEXT_ASSEMBLY_COMPLETED,
            agent_name="test-agent",
            run_id="run-1",
            metadata={
                "total_messages": 6,
                "estimated_tokens": 24,
                "source_counts": {
                    "system_prompt": 1,
                    "chat_history": 3,
                    "tool_result": 2,
                },
                "budget_status": {
                    "max_tokens_per_run": 100,
                    "consumed_tokens": 44,
                    "remaining_run_tokens": 56,
                    "estimated_remaining_after_context": 32,
                    "within_budget": True,
                },
            },
        ),
    ]

    evidence = extract_evidence(_run(), [], events)

    assert evidence.context_assembly_events_total == 2
    assert evidence.context_messages_total == 10
    assert evidence.context_estimated_tokens_total == 36
    assert evidence.context_estimated_tokens_max == 24
    assert evidence.context_budget_exceeded is False
    assert evidence.context_source_counts == {
        "system_prompt": 2,
        "semantic_memory": 2,
        "user_input": 1,
        "chat_history": 3,
        "tool_result": 2,
    }


def test_extract_evidence_counts_context_source_profile_changes() -> None:
    events = [
        AgentExecutionEvent(
            event_type=AgentExecutionEventType.CONTEXT_ASSEMBLY_COMPLETED,
            agent_name="test-agent",
            run_id="run-1",
            metadata={
                "total_messages": 3,
                "estimated_tokens": 12,
                "source_counts": {
                    "system_prompt": 1,
                    "semantic_memory": 1,
                    "user_input": 1,
                },
                "budget_status": {
                    "within_budget": True,
                },
            },
        ),
        AgentExecutionEvent(
            event_type=AgentExecutionEventType.CONTEXT_ASSEMBLY_COMPLETED,
            agent_name="test-agent",
            run_id="run-1",
            metadata={
                "total_messages": 3,
                "estimated_tokens": 12,
                "source_counts": {
                    "system_prompt": 1,
                    "semantic_memory": 1,
                    "user_input": 1,
                },
                "budget_status": {
                    "within_budget": True,
                },
            },
        ),
        AgentExecutionEvent(
            event_type=AgentExecutionEventType.CONTEXT_ASSEMBLY_COMPLETED,
            agent_name="test-agent",
            run_id="run-1",
            metadata={
                "total_messages": 4,
                "estimated_tokens": 18,
                "source_counts": {
                    "system_prompt": 1,
                    "semantic_memory": 1,
                    "tool_result": 1,
                    "user_input": 1,
                },
                "budget_status": {
                    "within_budget": True,
                },
            },
        ),
        AgentExecutionEvent(
            event_type=AgentExecutionEventType.CONTEXT_ASSEMBLY_COMPLETED,
            agent_name="test-agent",
            run_id="run-1",
            metadata={
                "total_messages": 5,
                "estimated_tokens": 22,
                "source_counts": {
                    "system_prompt": 1,
                    "semantic_memory": 1,
                    "tool_result": 2,
                    "user_input": 1,
                },
                "budget_status": {
                    "within_budget": True,
                },
            },
        ),
    ]

    evidence = extract_evidence(_run(), [], events)

    assert evidence.context_assembly_events_total == 4
    assert evidence.context_source_profile_changes == 1


def test_extract_evidence_ignores_source_count_only_changes() -> None:
    events = [
        AgentExecutionEvent(
            event_type=AgentExecutionEventType.CONTEXT_ASSEMBLY_COMPLETED,
            agent_name="test-agent",
            run_id="run-1",
            metadata={
                "source_counts": {
                    "system_prompt": 1,
                    "semantic_memory": 1,
                    "tool_result": 1,
                },
            },
        ),
        AgentExecutionEvent(
            event_type=AgentExecutionEventType.CONTEXT_ASSEMBLY_COMPLETED,
            agent_name="test-agent",
            run_id="run-1",
            metadata={
                "source_counts": {
                    "system_prompt": 1,
                    "semantic_memory": 1,
                    "tool_result": 2,
                },
            },
        ),
    ]

    evidence = extract_evidence(_run(), [], events)

    assert evidence.context_source_profile_changes == 0


def test_extract_evidence_ignores_malformed_context_diagnostics() -> None:
    events = [
        AgentExecutionEvent(
            event_type=AgentExecutionEventType.CONTEXT_ASSEMBLY_COMPLETED,
            agent_name="test-agent",
            run_id="run-1",
            metadata={
                "total_messages": -5,
                "estimated_tokens": -10,
                "source_counts": {
                    "system_prompt": -1,
                    "semantic_memory": "invalid",
                    123: 4,
                },
                "budget_status": {
                    "within_budget": "invalid",
                },
            },
        ),
        AgentExecutionEvent(
            event_type=AgentExecutionEventType.CONTEXT_ASSEMBLY_COMPLETED,
            agent_name="test-agent",
            run_id="run-1",
            metadata={
                "total_messages": 3,
                "estimated_tokens": 8,
                "source_counts": {
                    "system_prompt": 1,
                },
                "budget_status": {
                    "within_budget": False,
                },
            },
        ),
    ]

    evidence = extract_evidence(_run(), [], events)

    assert evidence.context_assembly_events_total == 2
    assert evidence.context_messages_total == 3
    assert evidence.context_estimated_tokens_total == 8
    assert evidence.context_estimated_tokens_max == 8
    assert evidence.context_budget_exceeded is True
    assert evidence.context_source_counts == {
        "system_prompt": 1,
    }


def test_extract_evidence_ignores_unrelated_event_diagnostics() -> None:
    events = [
        AgentExecutionEvent(
            event_type=AgentExecutionEventType.LLM_REQUESTED,
            agent_name="test-agent",
            run_id="run-1",
            metadata={
                "total_messages": 999,
                "estimated_tokens": 999,
                "source_counts": {
                    "system_prompt": 999,
                },
                "budget_status": {
                    "within_budget": False,
                },
            },
        ),
    ]

    evidence = extract_evidence(_run(), [], events)

    assert evidence.context_assembly_events_total == 0
    assert evidence.context_messages_total == 0
    assert evidence.context_estimated_tokens_total == 0
    assert evidence.context_estimated_tokens_max == 0
    assert evidence.context_budget_exceeded is False
    assert evidence.context_source_counts == {}


def test_extract_evidence_marks_rag_provenance_and_sources_available() -> None:
    steps = [
        _step(
            step_id="step-1",
            step_index=0,
            status=AgentRunStepStatus.COMPLETED,
            tool_name="rag.search",
            call_id="call-1",
            metadata={
                "rag_provenance": {
                    "retrieved_count": 2,
                    "sources": [
                        {"chunk_id": "chunk-1"},
                        {"chunk_id": "chunk-2"},
                    ],
                }
            },
        ),
    ]

    evidence = extract_evidence(_run(), steps, [])

    assert evidence.has_rag_provenance is True
    assert evidence.has_rag_sources_available is True


def test_extract_evidence_distinguishes_raw_rag_results_from_provenance() -> None:
    steps = [
        _step(
            step_id="step-1",
            step_index=0,
            status=AgentRunStepStatus.COMPLETED,
            tool_name="rag.search",
            call_id="call-1",
            output={
                "results": [
                    {"chunk_id": "chunk-1"},
                    {"chunk_id": "chunk-2"},
                ]
            },
        ),
    ]

    evidence = extract_evidence(_run(), steps, [])

    assert evidence.has_rag_provenance is False
    assert evidence.has_rag_sources_available is True


def test_extract_evidence_defaults_rag_provenance_signals_for_non_rag_runs() -> None:
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

    assert evidence.has_rag_provenance is False
    assert evidence.has_rag_sources_available is False


def test_extract_rag_source_texts_reads_raw_rag_content_transiently() -> None:
    steps = [
        _step(
            step_id="step-1",
            step_index=0,
            status=AgentRunStepStatus.COMPLETED,
            tool_name="rag.search",
            call_id="call-1",
            output={
                "query": "battery temperature",
                "results": [
                    {
                        "chunk_id": "chunk-1",
                        "content": "Vehicle V001 battery temperature reached 42 degrees Celsius.",
                    },
                    {
                        "chunk_id": "chunk-2",
                        "content": "Vehicle V001 telemetry was collected in September.",
                    },
                ],
            },
        ),
        _step(
            step_id="step-2",
            step_index=1,
            status=AgentRunStepStatus.COMPLETED,
            tool_name="search",
            call_id="call-2",
            output={
                "results": [
                    {
                        "content": "This must not be treated as RAG grounding evidence.",
                    }
                ]
            },
        ),
    ]

    source_texts = extract_rag_source_texts(steps)

    assert source_texts == [
        "Vehicle V001 battery temperature reached 42 degrees Celsius.",
        "Vehicle V001 telemetry was collected in September.",
    ]


def test_extract_rag_source_texts_ignores_missing_or_non_string_content() -> None:
    steps = [
        _step(
            step_id="step-1",
            step_index=0,
            status=AgentRunStepStatus.COMPLETED,
            tool_name="rag.search",
            output={
                "results": [
                    {"chunk_id": "chunk-1"},
                    {"chunk_id": "chunk-2", "content": ""},
                    {"chunk_id": "chunk-3", "content": "   "},
                    {"chunk_id": "chunk-4", "content": 42},
                    {
                        "chunk_id": "chunk-5",
                        "content": "Valid source text.",
                    },
                ]
            },
        )
    ]

    source_texts = extract_rag_source_texts(steps)

    assert source_texts == ["Valid source text."]


def test_extract_evidence_preserves_context_source_lineage_by_assembly() -> None:
    events = [
        AgentExecutionEvent(
            event_type=AgentExecutionEventType.CONTEXT_ASSEMBLY_COMPLETED,
            agent_name="test-agent",
            run_id="run-1",
            metadata={
                "source_lineage": [
                    {
                        "source_type": "system_prompt",
                        "item_id": None,
                        "score": None,
                        "reranker_score": None,
                    },
                    {
                        "source_type": "semantic_memory",
                        "item_id": "memory-1",
                        "score": 0.91,
                        "reranker_score": 0.87,
                        "provenance": {
                            "retrieval_method": "vector",
                            "rank": 1,
                            "source": "qdrant",
                        },
                    },
                    {
                        "source_type": "user_input",
                        "item_id": None,
                        "score": None,
                        "reranker_score": None,
                    },
                ],
            },
        ),
        AgentExecutionEvent(
            event_type=AgentExecutionEventType.CONTEXT_ASSEMBLY_COMPLETED,
            agent_name="test-agent",
            run_id="run-1",
            metadata={
                "source_lineage": [
                    {
                        "source_type": "system_prompt",
                        "item_id": None,
                        "score": None,
                        "reranker_score": None,
                    },
                    {
                        "source_type": "tool_result",
                        "item_id": "tool-1",
                        "score": None,
                        "reranker_score": None,
                    },
                ],
            },
        ),
    ]

    evidence = extract_evidence(_run(), [], events)

    assert evidence.context_source_lineage == (
        (
            {
                "source_type": "system_prompt",
                "item_id": None,
                "score": None,
                "reranker_score": None,
            },
            {
                "source_type": "semantic_memory",
                "item_id": "memory-1",
                "score": 0.91,
                "reranker_score": 0.87,
                "provenance": {
                    "retrieval_method": "vector",
                    "rank": 1,
                    "source": "qdrant",
                },
            },
            {
                "source_type": "user_input",
                "item_id": None,
                "score": None,
                "reranker_score": None,
            },
        ),
        (
            {
                "source_type": "system_prompt",
                "item_id": None,
                "score": None,
                "reranker_score": None,
            },
            {
                "source_type": "tool_result",
                "item_id": "tool-1",
                "score": None,
                "reranker_score": None,
            },
        ),
    )


def test_extract_evidence_ignores_malformed_context_source_lineage() -> None:
    events = [
        AgentExecutionEvent(
            event_type=AgentExecutionEventType.CONTEXT_ASSEMBLY_COMPLETED,
            agent_name="test-agent",
            run_id="run-1",
            metadata={
                "source_lineage": [
                    "invalid",
                    None,
                    123,
                    {
                        "source_type": "valid",
                        "provenance": "invalid",
                    },
                ],
            },
        ),
        AgentExecutionEvent(
            event_type=AgentExecutionEventType.CONTEXT_ASSEMBLY_COMPLETED,
            agent_name="test-agent",
            run_id="run-1",
            metadata={
                "source_lineage": "invalid",
            },
        ),
    ]

    evidence = extract_evidence(_run(), [], events)

    assert evidence.context_source_lineage == (
        (
            {
                "source_type": "valid",
                "provenance": "invalid",
            },
        ),
    )


def test_extract_evidence_defaults_context_source_lineage_for_missing_metadata() -> None:
    events = [
        AgentExecutionEvent(
            event_type=AgentExecutionEventType.CONTEXT_ASSEMBLY_COMPLETED,
            agent_name="test-agent",
            run_id="run-1",
            metadata={
                "total_messages": 2,
            },
        )
    ]

    evidence = extract_evidence(_run(), [], events)

    assert evidence.context_source_lineage == ()


def test_compute_evidence_fingerprint_is_deterministic() -> None:
    from ai_platform.agents.evaluation.evidence import compute_evidence_fingerprint

    steps = [
        _step(
            step_id="step-1",
            step_index=0,
            status=AgentRunStepStatus.COMPLETED,
            tool_name="search",
            call_id="call-1",
            input={"query": "vehicle", "filters": {"status": "active"}},
            output={"count": 3},
            metadata={"b": 2, "a": 1},
        )
    ]

    events = [
        AgentExecutionEvent(
            event_type=AgentExecutionEventType.TOOL_CALL_COMPLETED,
            agent_name="test-agent",
            run_id="run-1",
            tool_name="search",
            call_id="call-1",
            step_id="step-1",
            step_index=0,
            attempt=1,
            metadata={"b": 2, "a": 1},
        )
    ]

    assert compute_evidence_fingerprint(_run(), steps, events) == (
        compute_evidence_fingerprint(_run(), steps, events)
    )


def test_compute_evidence_fingerprint_ignores_mapping_key_order() -> None:
    from ai_platform.agents.evaluation.evidence import compute_evidence_fingerprint

    run_a = _run().model_copy(update={"metadata": {"a": 1, "nested": {"x": 1, "y": 2}}})
    run_b = _run().model_copy(update={"metadata": {"nested": {"y": 2, "x": 1}, "a": 1}})

    assert compute_evidence_fingerprint(run_a, [], []) == compute_evidence_fingerprint(
        run_b, [], []
    )


def test_compute_evidence_fingerprint_ignores_step_and_event_input_order() -> None:
    from ai_platform.agents.evaluation.evidence import compute_evidence_fingerprint

    steps = [
        _step(
            step_id="step-1",
            step_index=0,
            status=AgentRunStepStatus.COMPLETED,
        ),
        _step(
            step_id="step-2",
            step_index=1,
            status=AgentRunStepStatus.COMPLETED,
        ),
    ]

    events = [
        AgentExecutionEvent(
            event_type=AgentExecutionEventType.TOOL_CALL_COMPLETED,
            agent_name="test-agent",
            run_id="run-1",
            step_id="step-2",
            step_index=1,
            attempt=1,
            call_id="call-2",
        ),
        AgentExecutionEvent(
            event_type=AgentExecutionEventType.TOOL_CALL_COMPLETED,
            agent_name="test-agent",
            run_id="run-1",
            step_id="step-1",
            step_index=0,
            attempt=1,
            call_id="call-1",
        ),
    ]

    assert compute_evidence_fingerprint(_run(), steps, events) == (
        compute_evidence_fingerprint(_run(), list(reversed(steps)), list(reversed(events)))
    )


def test_compute_evidence_fingerprint_changes_for_material_evidence() -> None:
    from ai_platform.agents.evaluation.evidence import compute_evidence_fingerprint

    base = compute_evidence_fingerprint(_run(), [], [])

    changed_answer = _run().model_copy(update={"output": {"reply": "different answer"}})
    changed_answer_fingerprint = compute_evidence_fingerprint(changed_answer, [], [])

    changed_step = _step(
        step_id="step-1",
        step_index=0,
        status=AgentRunStepStatus.COMPLETED,
        input={"query": "different query"},
    )
    changed_step_fingerprint = compute_evidence_fingerprint(
        _run(),
        [changed_step],
        [],
    )

    changed_event = AgentExecutionEvent(
        event_type=AgentExecutionEventType.GOVERNANCE_DECISION,
        agent_name="test-agent",
        run_id="run-1",
        attempt=2,
        metadata={"decision": "deny"},
    )
    changed_event_fingerprint = compute_evidence_fingerprint(
        _run(),
        [],
        [changed_event],
    )

    assert changed_answer_fingerprint != base
    assert changed_step_fingerprint != base
    assert changed_event_fingerprint != base


def test_compute_evidence_fingerprint_normalizes_equivalent_timestamps() -> None:
    from ai_platform.agents.evaluation.evidence import compute_evidence_fingerprint

    run_utc = _run()
    ist = timezone(timedelta(hours=5, minutes=30))
    run_offset = run_utc.model_copy(
        update={
            "started_at": datetime(2026, 1, 1, 17, 30, tzinfo=ist),
            "completed_at": datetime(2026, 1, 1, 17, 30, 1, 250000, tzinfo=ist),
        }
    )

    assert compute_evidence_fingerprint(run_utc, [], []) == compute_evidence_fingerprint(
        run_offset, [], []
    )


def test_compute_evidence_fingerprint_rejects_non_finite_floats() -> None:
    from ai_platform.agents.evaluation.evidence import compute_evidence_fingerprint

    run = _run().model_copy(update={"metadata": {"invalid": float("nan")}})

    try:
        compute_evidence_fingerprint(run, [], [])
    except ValueError as exc:
        assert str(exc) == "evidence fingerprint cannot encode non-finite floats"
    else:
        raise AssertionError("Expected non-finite evidence float to be rejected")
