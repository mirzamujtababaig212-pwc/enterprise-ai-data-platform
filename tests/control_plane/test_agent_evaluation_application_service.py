from datetime import UTC, datetime, timedelta

import pytest

from ai_platform.agents.evaluation.evidence import compute_evidence_fingerprint
from ai_platform.agents.evaluation.policy import AgentEvaluationPolicy
from ai_platform.agents.evaluation.semantic_answer_evaluator import (
    SemanticAnswerEvaluation,
)
from ai_platform.agents.evaluation.semantic_grounding_evaluator import (
    SemanticGroundingEvaluation,
)
from app.control_plane.agent_evaluations.application_service import (
    AgentEvaluationApplicationService,
)
from app.control_plane.agent_evaluations.in_memory import (
    InMemoryAgentEvaluationRunsRepository,
)
from app.control_plane.agent_run_steps.models import (
    AgentRunStep,
    AgentRunStepStatus,
)
from app.control_plane.agent_runs.models import AgentRun, AgentRunStatus
from ai_platform.agents.observability import (
    AgentExecutionEvent,
    AgentExecutionEventType,
)


class FakeAgentRunRepository:
    def __init__(self, run: AgentRun | None) -> None:
        self.run = run
        self.requested_tenant_id: str | None = None

    def get_for_tenant(
        self,
        run_id: str,
        tenant_id: str,
    ) -> AgentRun | None:
        self.requested_tenant_id = tenant_id

        if self.run is None or self.run.run_id != run_id:
            return None

        if self.run.tenant_id != tenant_id:
            return None

        return self.run


class FakeAgentRunStepsRepository:
    def __init__(self, steps: list[AgentRunStep]) -> None:
        self.steps = steps
        self.requested_limit: int | None = None

    def list(
        self,
        run_id: str,
        *,
        status=None,
        limit: int = 100,
    ) -> list[AgentRunStep]:
        self.requested_limit = limit
        return [step for step in self.steps if step.run_id == run_id]


class FakeAgentRunEventsRepository:
    def __init__(self, events: list[AgentExecutionEvent]) -> None:
        self.events = events
        self.requested_limit: int | None = None

    def list(
        self,
        run_id: str,
        *,
        limit: int = 100,
    ) -> list[AgentExecutionEvent]:
        self.requested_limit = limit
        return [event for event in self.events if event.run_id == run_id]


def make_run(
    *,
    status: AgentRunStatus = AgentRunStatus.COMPLETED,
    tenant_id: str = "tenant-1",
    principal: str = "user-1",
    output=None,
) -> AgentRun:
    started_at = datetime(2026, 9, 24, 10, 0, tzinfo=UTC)
    completed_at = started_at + timedelta(seconds=2)

    return AgentRun(
        run_id="run-1",
        agent_name="vehicle-agent",
        tenant_id=tenant_id,
        principal=principal,
        status=status,
        started_at=started_at,
        completed_at=completed_at,
        metadata={"agent_version": "1.2.3"},
        output=output,
    )


def make_step(
    step_id: str,
    *,
    status: AgentRunStepStatus = AgentRunStepStatus.COMPLETED,
    tool_name: str | None = "lookup_vehicle",
    call_id: str | None = "call-1",
    failure_category: str | None = None,
    output=None,
    metadata=None,
) -> AgentRunStep:
    return AgentRunStep(
        run_id="run-1",
        step_id=step_id,
        step_index=0,
        step_type="tool",
        status=status,
        tool_name=tool_name,
        call_id=call_id,
        failure_category=failure_category,
        output=output,
        metadata=metadata or {},
    )


def make_governance_denial() -> AgentExecutionEvent:
    return AgentExecutionEvent(
        event_type=AgentExecutionEventType.GOVERNANCE_DECISION,
        agent_name="vehicle-agent",
        run_id="run-1",
        metadata={
            "governance_domain": "tool",
            "decision": "deny",
            "reason": "policy denied tool execution",
        },
    )


class FakeSemanticAnswerEvaluator:
    def __init__(
        self,
        *,
        score: float = 0.94,
        passed: bool = True,
    ) -> None:
        self.score = score
        self.passed = passed
        self.calls = []

    async def evaluate(
        self,
        *,
        actual_answer: str,
        expected_answer: str,
    ) -> SemanticAnswerEvaluation:
        self.calls.append(
            {
                "actual_answer": actual_answer,
                "expected_answer": expected_answer,
            }
        )
        return SemanticAnswerEvaluation(
            score=self.score,
            passed=self.passed,
            method="llm_judge_v1",
            evaluator_model="gpt-4.1-mini",
            evaluator_provider="openai",
        )


class FakeSemanticGroundingEvaluator:
    def __init__(
        self,
        *,
        score: float = 0.93,
        passed: bool = True,
    ) -> None:
        self.score = score
        self.passed = passed
        self.calls = []

    async def evaluate(
        self,
        *,
        answer_text: str,
        source_texts: list[str] | tuple[str, ...],
    ) -> SemanticGroundingEvaluation:
        self.calls.append(
            {
                "answer_text": answer_text,
                "source_texts": list(source_texts),
            }
        )
        return SemanticGroundingEvaluation(
            score=self.score,
            passed=self.passed,
            method="llm_grounding_judge_v1",
            evaluator_model="gpt-4.1-mini",
            evaluator_provider="openai",
        )


def make_service(
    *,
    run: AgentRun | None = None,
    steps: list[AgentRunStep] | None = None,
    events: list[AgentExecutionEvent] | None = None,
    semantic_evaluator=None,
    semantic_grounding_evaluator=None,
):
    run_repository = FakeAgentRunRepository(run or make_run())
    steps_repository = FakeAgentRunStepsRepository(steps or [])
    events_repository = FakeAgentRunEventsRepository(events or [])
    evaluation_repository = InMemoryAgentEvaluationRunsRepository()

    service = AgentEvaluationApplicationService(
        agent_run_repository=run_repository,
        agent_run_steps_repository=steps_repository,
        agent_run_events_repository=events_repository,
        evaluation_repository=evaluation_repository,
        semantic_evaluator=semantic_evaluator,
        semantic_grounding_evaluator=semantic_grounding_evaluator,
    )

    return (
        service,
        run_repository,
        steps_repository,
        events_repository,
        evaluation_repository,
    )


def default_policy() -> AgentEvaluationPolicy:
    return AgentEvaluationPolicy(
        max_execution_time_ms=5_000,
        max_steps_per_run=10,
        max_invalid_tool_calls=0,
        allow_governance_denials=False,
        require_task_completed=True,
        name="default-agent-quality",
    )


@pytest.mark.asyncio
async def test_evaluate_run_attaches_semantic_answer_evaluation():
    semantic_evaluator = FakeSemanticAnswerEvaluator()

    run = make_run(
        output={
            "reply": "The vehicle is powered by energy stored in a battery.",
        }
    )

    service, _, _, _, repository = make_service(
        run=run,
        semantic_evaluator=semantic_evaluator,
    )

    result = await service.evaluate_run(
        "run-1",
        tenant_id="tenant-1",
        principal="user-1",
        policy=default_policy(),
        expected_answer="Electric vehicles are powered by energy stored in batteries.",
    )

    assert semantic_evaluator.calls == [
        {
            "actual_answer": "The vehicle is powered by energy stored in a battery.",
            "expected_answer": ("Electric vehicles are powered by energy stored in batteries."),
        }
    ]

    assert result.answer_evaluation is not None
    assert result.answer_evaluation.evaluated is True
    assert result.answer_evaluation.exact_match is False
    assert result.answer_evaluation.semantic_evaluated is True
    assert result.answer_evaluation.semantic_score == 0.94
    assert result.answer_evaluation.semantic_passed is True
    assert result.answer_evaluation.semantic_method == "llm_judge_v1"
    assert result.answer_evaluation.evaluator_model == "gpt-4.1-mini"
    assert result.answer_evaluation.evaluator_provider == "openai"

    restored = repository.get(result.evaluation_run_id)
    assert restored == result
    assert restored is not None
    assert restored.answer_evaluation is not None
    assert restored.answer_evaluation.semantic_score == 0.94


@pytest.mark.asyncio
async def test_evaluate_run_does_not_invoke_semantic_answer_evaluator_without_expected_answer():
    semantic_evaluator = FakeSemanticAnswerEvaluator()

    service, _, _, _, _ = make_service(
        run=make_run(
            output={
                "reply": "The vehicle is powered by a battery.",
            }
        ),
        semantic_evaluator=semantic_evaluator,
    )

    result = await service.evaluate_run(
        "run-1",
        tenant_id="tenant-1",
        principal="user-1",
        policy=default_policy(),
    )

    assert semantic_evaluator.calls == []
    assert result.answer_evaluation is not None
    assert result.answer_evaluation.evaluated is False
    assert result.answer_evaluation.exact_match is None
    assert result.answer_evaluation.semantic_evaluated is False
    assert result.answer_evaluation.semantic_score is None
    assert result.answer_evaluation.semantic_passed is None


@pytest.mark.asyncio
async def test_semantic_answer_evaluation_does_not_change_deterministic_quality_gate():
    semantic_evaluator = FakeSemanticAnswerEvaluator(
        score=0.95,
        passed=True,
    )

    run = make_run(
        output={
            "reply": "The vehicle is powered by a battery.",
        }
    )

    service, _, _, _, _ = make_service(
        run=run,
        semantic_evaluator=semantic_evaluator,
    )

    policy = AgentEvaluationPolicy(
        require_task_completed=True,
        require_answer_match=True,
        name="exact-answer-quality",
    )

    result = await service.evaluate_run(
        "run-1",
        tenant_id="tenant-1",
        principal="user-1",
        policy=policy,
        expected_answer="The vehicle is powered by hydrogen.",
    )

    assert result.answer_evaluation is not None
    assert result.answer_evaluation.exact_match is False
    assert result.answer_evaluation.semantic_evaluated is True
    assert result.answer_evaluation.semantic_score == 0.95
    assert result.answer_evaluation.semantic_passed is True

    # Semantic answer evaluation is observational only.
    # The deterministic exact-match requirement remains authoritative.
    assert result.quality_gate.passed is False
    assert "Final answer did not match the expected answer." in (result.quality_gate.violations)


@pytest.mark.asyncio
async def test_evaluate_run_builds_and_persists_immutable_artifact():
    service, _, _, _, repository = make_service(
        steps=[
            make_step("step-1"),
            make_step("step-2", tool_name=None, call_id=None),
        ]
    )

    result = await service.evaluate_run(
        "run-1",
        tenant_id="tenant-1",
        principal="user-1",
        policy=default_policy(),
    )

    assert result.lineage.evaluated_run_id == "run-1"
    assert result.lineage.agent_name == "vehicle-agent"
    assert result.lineage.agent_version == "1.2.3"
    assert result.lineage.tenant_id == "tenant-1"
    assert result.metrics.steps_total == 2
    assert result.metrics.tool_calls_total == 1
    assert result.metrics.tool_calls_successful == 1
    assert result.metrics.tool_calls_failed == 0
    assert result.metrics.invalid_tool_calls == 0
    assert result.metrics.governance_denials == 0
    assert result.metrics.task_completed is True
    assert result.passed is True
    assert result.context_quality is not None
    assert result.context_quality.budget_compliant is None
    assert result.context_quality.retrieval_evidence_present is None
    assert result.context_quality.assemblies_total == 0
    assert repository.get(result.evaluation_run_id) == result


@pytest.mark.asyncio
async def test_evaluate_run_counts_governance_denials():
    service, _, _, _, _ = make_service(events=[make_governance_denial()])

    result = await service.evaluate_run(
        "run-1",
        tenant_id="tenant-1",
        principal="user-1",
        policy=default_policy(),
    )

    assert result.metrics.governance_denials == 1
    assert result.passed is False
    assert "governance denial" in result.quality_gate.violations[0]


@pytest.mark.asyncio
async def test_evaluate_run_preserves_failed_tool_metrics():
    service, _, _, _, _ = make_service(
        steps=[
            make_step(
                "step-1",
                status=AgentRunStepStatus.FAILED,
                failure_category="execution_error",
            ),
        ]
    )

    result = await service.evaluate_run(
        "run-1",
        tenant_id="tenant-1",
        principal="user-1",
        policy=default_policy(),
    )

    assert result.metrics.tool_calls_total == 1
    assert result.metrics.tool_calls_successful == 0
    assert result.metrics.tool_calls_failed == 1


@pytest.mark.asyncio
async def test_evaluate_run_applies_rag_score_thresholds_to_durable_evidence():
    service, _, _, _, repository = make_service(
        steps=[
            AgentRunStep(
                run_id="run-1",
                step_id="step-rag",
                step_index=0,
                step_type="tool",
                status=AgentRunStepStatus.COMPLETED,
                tool_name="rag.search",
                call_id="call-rag-1",
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
    )

    policy = AgentEvaluationPolicy(
        min_retrieval_score=0.70,
        min_reranker_score=0.90,
        name="rag-quality-v1",
    )

    result = await service.evaluate_run(
        "run-1",
        tenant_id="tenant-1",
        principal="user-1",
        policy=policy,
    )

    assert result.metrics.retrieval_score_avg == pytest.approx((0.72 + 0.61 + 0.83) / 3)
    assert result.metrics.reranker_score_avg == pytest.approx((0.91 + 0.88 + 0.95) / 3)
    assert result.passed is True
    assert result.quality_gate.violations == ()
    assert repository.get(result.evaluation_run_id) == result


@pytest.mark.asyncio
async def test_evaluate_run_rejects_rag_score_thresholds_from_durable_evidence():
    service, _, _, _, repository = make_service(
        steps=[
            AgentRunStep(
                run_id="run-1",
                step_id="step-rag",
                step_index=0,
                step_type="tool",
                status=AgentRunStepStatus.COMPLETED,
                tool_name="rag.search",
                call_id="call-rag-1",
                metadata={
                    "rag_provenance": {
                        "retrieved_count": 3,
                        "sources": [
                            {
                                "chunk_id": "chunk-1",
                                "retrieval_score": 0.52,
                                "reranker_score": 0.91,
                            },
                            {
                                "chunk_id": "chunk-2",
                                "retrieval_score": 0.61,
                                "reranker_score": 0.88,
                            },
                            {
                                "chunk_id": "chunk-3",
                                "retrieval_score": 0.63,
                                "reranker_score": 0.95,
                            },
                        ],
                    }
                },
            ),
        ]
    )

    policy = AgentEvaluationPolicy(
        min_retrieval_score=0.70,
        min_reranker_score=0.90,
        name="rag-quality-v1",
    )

    result = await service.evaluate_run(
        "run-1",
        tenant_id="tenant-1",
        principal="user-1",
        policy=policy,
    )

    assert result.metrics.retrieval_score_avg == pytest.approx((0.52 + 0.61 + 0.63) / 3)
    assert result.metrics.reranker_score_avg == pytest.approx((0.91 + 0.88 + 0.95) / 3)
    assert result.passed is False
    assert any(
        "Average retrieval score" in violation for violation in result.quality_gate.violations
    )
    assert repository.get(result.evaluation_run_id) == result


@pytest.mark.asyncio
async def test_evaluate_run_rejects_missing_run():
    service, _, _, _, _ = make_service(run=None)

    with pytest.raises(
        LookupError,
        match="agent run not found: missing",
    ):
        await service.evaluate_run(
            "missing",
            tenant_id="tenant-1",
            principal="user-1",
            policy=default_policy(),
        )


@pytest.mark.asyncio
async def test_evaluate_run_enforces_principal_authorization():
    service, _, _, _, _ = make_service(
        run=make_run(principal="different-user"),
    )

    with pytest.raises(
        PermissionError,
        match="principal is not authorized",
    ):
        await service.evaluate_run(
            "run-1",
            tenant_id="tenant-1",
            principal="user-1",
            policy=default_policy(),
        )


@pytest.mark.asyncio
async def test_evaluate_run_uses_bounded_repository_reads():
    service, _, steps_repository, events_repository, _ = make_service()

    await service.evaluate_run(
        "run-1",
        tenant_id="tenant-1",
        principal="user-1",
        policy=default_policy(),
    )

    assert steps_repository.requested_limit == 10_000
    assert events_repository.requested_limit == 10_000


@pytest.mark.asyncio
async def test_evaluate_run_attaches_semantic_grounding_aggregates():
    semantic_evaluator = FakeSemanticGroundingEvaluator()

    run = make_run(output={"reply": "The vehicle battery temperature reached 42 degrees Celsius."})
    rag_step = make_step(
        "step-1",
        tool_name="rag.search",
        call_id="rag-call-1",
        output={
            "results": [
                {
                    "chunk_id": "chunk-1",
                    "content": (
                        "Vehicle V001 telemetry shows the battery temperature "
                        "reached 42 degrees Celsius."
                    ),
                },
                {
                    "chunk_id": "chunk-2",
                    "content": "Vehicle V001 telemetry was collected in September.",
                },
            ]
        },
    )

    service, _, _, _, repository = make_service(
        run=run,
        steps=[rag_step],
        semantic_grounding_evaluator=semantic_evaluator,
    )

    result = await service.evaluate_run(
        "run-1",
        tenant_id="tenant-1",
        principal="user-1",
        policy=default_policy(),
    )

    assert semantic_evaluator.calls == [
        {
            "answer_text": ("The vehicle battery temperature reached 42 degrees Celsius."),
            "source_texts": [
                (
                    "Vehicle V001 telemetry shows the battery temperature "
                    "reached 42 degrees Celsius."
                ),
                "Vehicle V001 telemetry was collected in September.",
            ],
        }
    ]

    assert result.metrics.semantic_grounding_evaluated is True
    assert result.metrics.semantic_grounding_score == 0.93
    assert result.metrics.semantic_grounding_passed is True
    assert result.metrics.semantic_grounding_method == "llm_grounding_judge_v1"
    assert result.metrics.semantic_grounding_evaluator_model == "gpt-4.1-mini"
    assert result.metrics.semantic_grounding_evaluator_provider == "openai"

    assert result.metrics.grounding_evaluated is True
    assert result.metrics.grounding_supported is True
    assert result.metrics.grounding_support_ratio == 1.0
    assert result.metrics.grounding_supported_sources_total == 1
    assert result.metrics.grounding_source_candidates_total == 2
    assert result.metrics.grounding_method == "lexical_sentence_support_v1"

    restored = repository.get(result.evaluation_run_id)
    assert restored == result


@pytest.mark.asyncio
async def test_semantic_grounding_does_not_change_deterministic_quality_gate():
    semantic_evaluator = FakeSemanticGroundingEvaluator(
        score=0.20,
        passed=False,
    )

    run = make_run(output={"reply": "The vehicle battery temperature reached 99 degrees Celsius."})
    rag_step = make_step(
        "step-1",
        tool_name="rag.search",
        call_id="rag-call-1",
        output={
            "results": [
                {
                    "chunk_id": "chunk-1",
                    "content": (
                        "Vehicle V001 telemetry shows the battery temperature "
                        "reached 42 degrees Celsius."
                    ),
                }
            ]
        },
    )

    service, _, _, _, _ = make_service(
        run=run,
        steps=[rag_step],
        semantic_grounding_evaluator=semantic_evaluator,
    )

    result = await service.evaluate_run(
        "run-1",
        tenant_id="tenant-1",
        principal="user-1",
        policy=default_policy(),
    )

    assert result.metrics.semantic_grounding_evaluated is True
    assert result.metrics.semantic_grounding_score == 0.20
    assert result.metrics.semantic_grounding_passed is False

    # Semantic grounding is observational only; the existing deterministic
    # quality gate remains authoritative.
    assert result.quality_gate.passed is True
    assert result.quality_gate.violations == ()


@pytest.mark.asyncio
async def test_semantic_grounding_policy_threshold_controls_quality_gate():
    semantic_evaluator = FakeSemanticGroundingEvaluator(
        score=0.79,
        passed=False,
    )

    run = make_run(
        output={
            "reply": "The vehicle battery temperature reached 42 degrees Celsius.",
        }
    )
    rag_step = make_step(
        "step-1",
        tool_name="rag.search",
        call_id="rag-call-1",
        output={
            "results": [
                {
                    "chunk_id": "chunk-1",
                    "content": (
                        "Vehicle V001 telemetry shows the battery temperature "
                        "reached 42 degrees Celsius."
                    ),
                }
            ]
        },
    )

    service, _, _, _, repository = make_service(
        run=run,
        steps=[rag_step],
        semantic_grounding_evaluator=semantic_evaluator,
    )

    result = await service.evaluate_run(
        "run-1",
        tenant_id="tenant-1",
        principal="user-1",
        policy=AgentEvaluationPolicy(
            min_semantic_grounding_score=0.80,
        ),
    )

    assert semantic_evaluator.calls == [
        {
            "answer_text": ("The vehicle battery temperature reached 42 degrees Celsius."),
            "source_texts": [
                (
                    "Vehicle V001 telemetry shows the battery temperature "
                    "reached 42 degrees Celsius."
                )
            ],
        }
    ]

    assert result.metrics.semantic_grounding_evaluated is True
    assert result.metrics.semantic_grounding_score == 0.79
    assert result.metrics.semantic_grounding_passed is False

    assert result.quality_gate.passed is False
    assert result.quality_gate.violations == (
        "Semantic grounding score (0.79) was below minimum threshold (0.80).",
    )

    restored = repository.get(result.evaluation_run_id)
    assert restored == result
    assert restored.policy.min_semantic_grounding_score == 0.80


@pytest.mark.asyncio
async def test_evaluate_run_leaves_semantic_grounding_unset_without_answer_or_sources():
    semantic_evaluator = FakeSemanticGroundingEvaluator()

    service, _, _, _, _ = make_service(
        run=make_run(output={"reply": "An answer exists, but no RAG source exists."}),
        steps=[],
        semantic_grounding_evaluator=semantic_evaluator,
    )

    result = await service.evaluate_run(
        "run-1",
        tenant_id="tenant-1",
        principal="user-1",
        policy=default_policy(),
    )

    assert semantic_evaluator.calls == []
    assert result.metrics.semantic_grounding_evaluated is False
    assert result.metrics.semantic_grounding_score is None
    assert result.metrics.semantic_grounding_passed is None
    assert result.metrics.semantic_grounding_method is None
    assert result.metrics.semantic_grounding_evaluator_model is None
    assert result.metrics.semantic_grounding_evaluator_provider is None


@pytest.mark.asyncio
async def test_evaluate_run_persists_context_quality_from_context_assembly_events():
    context_event = AgentExecutionEvent(
        event_type=AgentExecutionEventType.CONTEXT_ASSEMBLY_COMPLETED,
        agent_name="vehicle-agent",
        run_id="run-1",
        metadata={
            "total_messages": 7,
            "source_counts": {
                "system_prompt": 1,
                "semantic_memory": 2,
                "chat_history": 1,
                "user_input": 1,
                "tool_result": 2,
            },
            "estimated_tokens": 128,
            "budget_status": {
                "within_budget": True,
            },
        },
    )

    service, _, _, _, repository = make_service(
        events=[context_event],
    )

    result = await service.evaluate_run(
        "run-1",
        tenant_id="tenant-1",
        principal="user-1",
        policy=default_policy(),
    )

    assert result.context_quality is not None
    assert result.context_quality.budget_compliant is True
    assert result.context_quality.retrieval_evidence_present is True
    assert result.context_quality.has_semantic_memory_sources is True
    assert result.context_quality.has_episodic_memory_sources is False
    assert result.context_quality.has_chat_history_sources is True
    assert result.context_quality.has_tool_result_sources is True
    assert result.context_quality.assemblies_total == 1
    assert result.context_quality.messages_total == 7
    assert result.context_quality.estimated_tokens_total == 128
    assert result.context_quality.estimated_tokens_max == 128
    assert result.context_quality.source_counts["semantic_memory"] == 2
    assert result.quality_gate.passed is True
    restored = repository.get(result.evaluation_run_id)
    assert restored is not None
    assert restored.context_quality is not None
    assert result.context_quality is not None
    assert (
        restored.context_quality.minimum_estimated_remaining_after_context
        == result.context_quality.minimum_estimated_remaining_after_context
    )
    assert repository.get(result.evaluation_run_id) == result


@pytest.mark.asyncio
async def test_evaluate_run_with_diagnostics_preserves_transient_grounding_attribution():
    service, run_repository, steps_repository, events_repository, repository = make_service(
        run=make_run(
            output={
                "answer": "Vehicle V001 traveled at 62 miles per hour.",
            }
        ),
        steps=[
            AgentRunStep(
                run_id="run-1",
                step_id="step-rag",
                step_index=0,
                step_type="tool",
                status=AgentRunStepStatus.COMPLETED,
                tool_name="rag.search",
                call_id="call-rag-1",
                output={
                    "results": [
                        {
                            "content": (
                                "Vehicle V001 traveled at 62 miles per hour "
                                "during the recorded interval."
                            ),
                            "chunk_id": "chunk-v001-001",
                            "document_id": "doc-v001",
                            "retrieval_score": 0.91,
                            "reranker_score": 0.95,
                        },
                    ],
                },
                metadata={
                    "rag_provenance": {
                        "retrieved_count": 1,
                        "sources": [
                            {
                                "chunk_id": "chunk-v001-001",
                                "document_id": "doc-v001",
                                "retrieval_score": 0.91,
                                "reranker_score": 0.95,
                            }
                        ],
                    }
                },
            ),
        ],
    )

    result = await service.evaluate_run_with_diagnostics(
        "run-1",
        tenant_id="tenant-1",
        principal="user-1",
        policy=default_policy(),
    )

    assert result.evaluation_run is repository.get(result.evaluation_run.evaluation_run_id)

    attributions = result.diagnostics.grounding_attributions

    assert len(attributions) == 1
    assert attributions[0].claim_index == 0
    assert attributions[0].supported is True
    assert attributions[0].supporting_source_indexes == (0,)
    assert attributions[0].supporting_source_ids == ("chunk-v001-001",)

    persisted = repository.get(result.evaluation_run.evaluation_run_id)
    assert persisted is not None

    durable_run = run_repository.run
    assert durable_run is not None

    expected_fingerprint = compute_evidence_fingerprint(
        durable_run,
        steps_repository.steps,
        events_repository.events,
    )

    assert persisted.lineage.evidence_fingerprint == expected_fingerprint
    assert len(persisted.lineage.evidence_fingerprint) == 64

    assert persisted.metrics.grounding_evaluated is True
    assert persisted.metrics.grounding_supported is True
    assert persisted.metrics.grounding_support_ratio == 1.0

    persisted_payload = persisted.as_dict()
    assert "grounding_attributions" not in persisted_payload
    assert "attributions" not in persisted_payload["metrics"]


@pytest.mark.asyncio
async def test_get_run_diagnostics_recomputes_transient_grounding_attribution():
    service, _, steps_repository, events_repository, repository = make_service(
        run=make_run(
            output={
                "answer": "Vehicle V001 traveled at 62 miles per hour.",
            }
        ),
        steps=[
            AgentRunStep(
                run_id="run-1",
                step_id="step-rag",
                step_index=0,
                step_type="tool",
                status=AgentRunStepStatus.COMPLETED,
                tool_name="rag.search",
                call_id="call-rag-1",
                output={
                    "results": [
                        {
                            "content": (
                                "Vehicle V001 traveled at 62 miles per hour "
                                "during the recorded interval."
                            ),
                            "chunk_id": "chunk-v001-001",
                            "document_id": "doc-v001",
                            "retrieval_score": 0.91,
                            "reranker_score": 0.95,
                        },
                    ],
                },
            ),
        ],
    )

    diagnostics = service.get_run_diagnostics(
        "run-1",
        tenant_id="tenant-1",
        principal="user-1",
    )

    assert len(diagnostics.grounding_attributions) == 1

    attribution = diagnostics.grounding_attributions[0]
    assert attribution.claim_index == 0
    assert attribution.claim_text == ("Vehicle V001 traveled at 62 miles per hour.")
    assert attribution.supported is True
    assert attribution.supporting_source_indexes == (0,)
    assert attribution.supporting_source_ids == ("chunk-v001-001",)

    assert steps_repository.requested_limit == 10_000
    assert events_repository.requested_limit == 10_000
    assert (
        repository.list(
            evaluated_run_id="run-1",
            tenant_id="tenant-1",
            limit=100,
        )
        == []
    )


def test_get_run_diagnostics_rejects_missing_run():
    service, _, _, _, _ = make_service(run=None)

    with pytest.raises(
        LookupError,
        match="agent run not found: missing",
    ):
        service.get_run_diagnostics(
            "missing",
            tenant_id="tenant-1",
            principal="user-1",
        )


def test_get_run_diagnostics_enforces_principal_authorization():
    service, _, _, _, _ = make_service(
        run=make_run(principal="different-user"),
    )

    with pytest.raises(
        PermissionError,
        match="principal is not authorized",
    ):
        service.get_run_diagnostics(
            "run-1",
            tenant_id="tenant-1",
            principal="user-1",
        )
