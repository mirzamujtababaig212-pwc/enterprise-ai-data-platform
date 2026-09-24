from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from ai_platform.agents.evaluation.evaluator import AgentEvaluator
from ai_platform.agents.evaluation.evidence import extract_evidence
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
    ) -> None:
        self._agent_run_repository = agent_run_repository
        self._agent_run_steps_repository = agent_run_steps_repository
        self._agent_run_events_repository = agent_run_events_repository
        self._evaluation_repository = evaluation_repository
        self._evaluator = evaluator or AgentEvaluator()

    def evaluate_run(
        self,
        run_id: str,
        *,
        tenant_id: str,
        principal: str,
        policy: AgentEvaluationPolicy,
    ) -> AgentEvaluationRun:
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

        metrics, quality_gate = self._evaluator.evaluate_run(
            evidence,
            policy,
        )

        evaluation_run = AgentEvaluationRun(
            evaluation_run_id=str(uuid4()),
            created_at=datetime.now(UTC),
            lineage=AgentEvaluationLineage(
                evaluated_run_id=run.run_id,
                agent_name=run.agent_name,
                agent_version=run.metadata.get("agent_version"),
                tenant_id=run.tenant_id,
            ),
            metrics=metrics,
            policy=policy,
            quality_gate=quality_gate,
        )

        return self._evaluation_repository.save(evaluation_run)

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
