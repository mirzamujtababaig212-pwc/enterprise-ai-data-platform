from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from uuid import uuid4

from ai_platform.agents.evaluation.diagnostics import (
    AgentEvaluationDiagnostics,
    AgentEvaluationResult,
)
from ai_platform.agents.evaluation.evaluator import AgentEvaluator
from ai_platform.agents.evaluation.semantic_answer_evaluator import (
    SemanticAnswerEvaluator,
)
from ai_platform.agents.evaluation.semantic_grounding_evaluator import (
    SemanticGroundingEvaluator,
)
from ai_platform.agents.evaluation.evidence import (
    extract_evidence,
    extract_rag_evidence_sources,
)
from ai_platform.agents.evaluation.policy import AgentEvaluationPolicy
from ai_platform.agents.evaluation.run import (
    AgentEvaluationLineage,
    AgentEvaluationRun,
)
from app.control_plane.agent_evaluations.repository import (
    AgentEvaluationRunsRepository,
)
from app.control_plane.agent_run_events.repository import AgentRunEventsRepository
from app.control_plane.agent_run_steps.repository import AgentRunStepsRepository
from app.control_plane.agent_runs.models import AgentRun
from app.control_plane.agent_runs.repository import AgentRunRepository


class AgentEvaluationApplicationService:
    """Application boundary for evaluating durable agent executions."""

    def __init__(
        self,
        *,
        agent_run_repository: AgentRunRepository,
        agent_run_steps_repository: AgentRunStepsRepository,
        agent_run_events_repository: AgentRunEventsRepository,
        evaluation_repository: AgentEvaluationRunsRepository,
        evaluator: AgentEvaluator | None = None,
        semantic_evaluator: SemanticAnswerEvaluator | None = None,
        semantic_grounding_evaluator: SemanticGroundingEvaluator | None = None,
    ) -> None:
        self._agent_run_repository = agent_run_repository
        self._agent_run_steps_repository = agent_run_steps_repository
        self._agent_run_events_repository = agent_run_events_repository
        self._evaluation_repository = evaluation_repository
        self._evaluator = evaluator or AgentEvaluator()
        self._semantic_evaluator = semantic_evaluator
        self._semantic_grounding_evaluator = semantic_grounding_evaluator

    async def evaluate_run(
        self,
        run_id: str,
        *,
        tenant_id: str,
        principal: str,
        policy: AgentEvaluationPolicy,
        expected_answer: str | None = None,
    ) -> AgentEvaluationRun:
        result = await self.evaluate_run_with_diagnostics(
            run_id,
            tenant_id=tenant_id,
            principal=principal,
            policy=policy,
            expected_answer=expected_answer,
        )
        return result.evaluation_run

    async def evaluate_run_with_diagnostics(
        self,
        run_id: str,
        *,
        tenant_id: str,
        principal: str,
        policy: AgentEvaluationPolicy,
        expected_answer: str | None = None,
    ) -> AgentEvaluationResult:
        if not run_id.strip():
            raise ValueError("run_id must not be empty.")

        if not tenant_id.strip():
            raise ValueError("tenant_id must not be empty.")

        if not principal.strip():
            raise ValueError("principal must not be empty.")

        run = self._agent_run_repository.get_for_tenant(
            run_id,
            tenant_id,
        )

        if run is None:
            raise LookupError(f"agent run not found: {run_id}")

        self._authorize_run(run, tenant_id=tenant_id, principal=principal)

        steps = self._agent_run_steps_repository.list(
            run_id,
            limit=10_000,
        )
        events = self._agent_run_events_repository.list(
            run_id,
            limit=10_000,
        )

        evidence = extract_evidence(
            run,
            steps,
            events,
        )

        answer_evaluation = self._evaluator.evaluate_answer(
            evidence,
            expected_answer=expected_answer,
        )

        if (
            self._semantic_evaluator is not None
            and expected_answer is not None
            and evidence.final_answer_text is not None
        ):
            semantic_evaluation = await self._semantic_evaluator.evaluate(
                actual_answer=evidence.final_answer_text,
                expected_answer=expected_answer,
            )
            answer_evaluation = replace(
                answer_evaluation,
                semantic_evaluated=True,
                semantic_score=semantic_evaluation.score,
                semantic_passed=semantic_evaluation.passed,
                semantic_method=semantic_evaluation.method,
                evaluator_model=semantic_evaluation.evaluator_model,
                evaluator_provider=semantic_evaluation.evaluator_provider,
            )

        rag_sources = extract_rag_evidence_sources(steps)

        grounding_evaluation = self._evaluator.evaluate_grounding(
            evidence,
            sources=rag_sources,
        )

        metrics, quality_gate = self._evaluator.evaluate_run(
            evidence,
            policy,
            answer_evaluation=answer_evaluation,
            grounding_evaluation=grounding_evaluation,
        )

        if (
            self._semantic_grounding_evaluator is not None
            and evidence.final_answer_text is not None
            and rag_sources
        ):
            semantic_grounding_evaluation = await self._semantic_grounding_evaluator.evaluate(
                answer_text=evidence.final_answer_text,
                source_texts=[source.content for source in rag_sources],
            )
            metrics = replace(
                metrics,
                semantic_grounding_evaluated=True,
                semantic_grounding_score=semantic_grounding_evaluation.score,
                semantic_grounding_passed=semantic_grounding_evaluation.passed,
                semantic_grounding_method=semantic_grounding_evaluation.method,
                semantic_grounding_evaluator_model=(semantic_grounding_evaluation.evaluator_model),
                semantic_grounding_evaluator_provider=(
                    semantic_grounding_evaluation.evaluator_provider
                ),
            )

        context_quality = self._evaluator.evaluate_context(evidence)

        evaluation_run = AgentEvaluationRun(
            evaluation_run_id=str(uuid4()),
            created_at=datetime.now(UTC),
            lineage=AgentEvaluationLineage(
                evaluated_run_id=run.run_id,
                agent_name=run.agent_name,
                agent_version=run.metadata.get("agent_version"),
                tenant_id=run.tenant_id,
                effective_model=evidence.effective_model,
                effective_provider=evidence.effective_provider,
                model_policy_id=evidence.model_policy_id,
                model_policy_version=evidence.model_policy_version,
            ),
            metrics=metrics,
            policy=policy,
            quality_gate=quality_gate,
            answer_evaluation=answer_evaluation,
            context_quality=context_quality,
        )

        persisted_evaluation = self._evaluation_repository.save(evaluation_run)

        return AgentEvaluationResult(
            evaluation_run=persisted_evaluation,
            diagnostics=AgentEvaluationDiagnostics(
                grounding_attributions=grounding_evaluation.attributions,
            ),
        )

    def list_evaluations(
        self,
        run_id: str,
        *,
        tenant_id: str,
        principal: str,
        limit: int = 100,
    ) -> list[AgentEvaluationRun]:
        if not run_id.strip():
            raise ValueError("run_id must not be empty.")

        if not tenant_id.strip():
            raise ValueError("tenant_id must not be empty.")

        if not principal.strip():
            raise ValueError("principal must not be empty.")

        if limit <= 0:
            raise ValueError("limit must be greater than zero.")

        run = self._agent_run_repository.get_for_tenant(
            run_id,
            tenant_id,
        )

        if run is None:
            raise LookupError(f"agent run not found: {run_id}")

        self._authorize_run(
            run,
            tenant_id=tenant_id,
            principal=principal,
        )

        return self._evaluation_repository.list(
            evaluated_run_id=run_id,
            tenant_id=tenant_id,
            limit=limit,
        )

    @staticmethod
    def _authorize_run(
        run: AgentRun,
        *,
        tenant_id: str,
        principal: str,
    ) -> None:
        if run.tenant_id != tenant_id:
            raise PermissionError("agent run does not belong to tenant.")

        if run.principal != principal:
            raise PermissionError("principal is not authorized to access agent run.")
