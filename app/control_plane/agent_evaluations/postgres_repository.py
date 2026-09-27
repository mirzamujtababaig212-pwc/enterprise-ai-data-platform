from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ai_platform.agents.evaluation.answer_evaluation import AgentAnswerEvaluation
from ai_platform.agents.evaluation.models import AgentEvaluationMetrics
from ai_platform.agents.evaluation.policy import (
    AgentEvaluationPolicy,
    AgentQualityGateResult,
)
from ai_platform.agents.evaluation.run import (
    AgentEvaluationLineage,
    AgentEvaluationRun,
)
from app.control_plane.agent_evaluations.repository import (
    DuplicateAgentEvaluationRunError,
)
from app.control_plane.persistence.models import AgentEvaluationRunRecord


class PostgreSQLAgentEvaluationRunsRepository:
    """Durable repository for immutable agent execution evaluation artifacts."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def close(self) -> None:
        self._session.close()

    def save(
        self,
        run: AgentEvaluationRun,
        *,
        commit: bool = True,
    ) -> AgentEvaluationRun:
        existing = self._session.scalar(
            select(AgentEvaluationRunRecord).where(
                AgentEvaluationRunRecord.evaluation_run_id == run.evaluation_run_id
            )
        )

        if existing is not None:
            raise DuplicateAgentEvaluationRunError(
                f"agent evaluation run already exists: " f"{run.evaluation_run_id}"
            )

        record = AgentEvaluationRunRecord(
            evaluation_run_id=run.evaluation_run_id,
            created_at=run.created_at,
            evaluated_run_id=run.lineage.evaluated_run_id,
            agent_name=run.lineage.agent_name,
            agent_version=run.lineage.agent_version,
            tenant_id=run.lineage.tenant_id,
            passed=run.passed,
            lineage=run.lineage.as_dict(),
            metrics=run.metrics.as_dict(),
            policy=run.policy.as_dict(),
            quality_gate=run.quality_gate.as_dict(),
            answer_evaluation=(
                run.answer_evaluation.as_dict() if run.answer_evaluation is not None else None
            ),
        )

        try:
            self._session.add(record)
            self._session.flush()

            if commit:
                self._session.commit()

        except IntegrityError as exc:
            if commit:
                self._session.rollback()

            raise DuplicateAgentEvaluationRunError(
                f"agent evaluation run already exists: " f"{run.evaluation_run_id}"
            ) from exc

        except Exception:
            if commit:
                self._session.rollback()
            raise

        return run

    def get(
        self,
        evaluation_run_id: str,
    ) -> AgentEvaluationRun | None:
        record = self._session.scalar(
            select(AgentEvaluationRunRecord).where(
                AgentEvaluationRunRecord.evaluation_run_id == evaluation_run_id
            )
        )

        if record is None:
            return None

        return self._to_domain(record)

    def list(
        self,
        *,
        evaluated_run_id: str | None = None,
        tenant_id: str | None = None,
        limit: int = 100,
    ) -> list[AgentEvaluationRun]:
        if limit <= 0:
            raise ValueError("limit must be greater than zero.")

        statement = (
            select(AgentEvaluationRunRecord)
            .order_by(
                AgentEvaluationRunRecord.created_at.desc(),
                AgentEvaluationRunRecord.evaluation_run_id.desc(),
            )
            .limit(limit)
        )

        if evaluated_run_id is not None:
            statement = statement.where(
                AgentEvaluationRunRecord.evaluated_run_id == evaluated_run_id
            )

        if tenant_id is not None:
            statement = statement.where(AgentEvaluationRunRecord.tenant_id == tenant_id)

        records = self._session.scalars(statement).all()

        return [self._to_domain(record) for record in records]

    def count(
        self,
        *,
        evaluated_run_id: str | None = None,
        tenant_id: str | None = None,
    ) -> int:
        statement = select(func.count(AgentEvaluationRunRecord.evaluation_run_id))

        if evaluated_run_id is not None:
            statement = statement.where(
                AgentEvaluationRunRecord.evaluated_run_id == evaluated_run_id
            )

        if tenant_id is not None:
            statement = statement.where(AgentEvaluationRunRecord.tenant_id == tenant_id)

        return int(self._session.scalar(statement) or 0)

    @staticmethod
    def _to_domain(
        record: AgentEvaluationRunRecord,
    ) -> AgentEvaluationRun:
        lineage_data = dict(record.lineage)
        metrics_data = dict(record.metrics)
        policy_data = dict(record.policy)
        quality_gate_data = dict(record.quality_gate)
        answer_evaluation_data = (
            dict(record.answer_evaluation) if record.answer_evaluation is not None else None
        )

        return AgentEvaluationRun(
            evaluation_run_id=record.evaluation_run_id,
            created_at=record.created_at,
            lineage=AgentEvaluationLineage(
                evaluated_run_id=lineage_data["evaluated_run_id"],
                agent_name=lineage_data["agent_name"],
                agent_version=lineage_data.get("agent_version"),
                tenant_id=lineage_data.get("tenant_id"),
                effective_model=lineage_data.get("effective_model"),
                effective_provider=lineage_data.get("effective_provider"),
                model_policy_id=lineage_data.get("model_policy_id"),
                model_policy_version=lineage_data.get("model_policy_version"),
            ),
            metrics=AgentEvaluationMetrics(
                execution_time_ms=metrics_data["execution_time_ms"],
                steps_total=metrics_data["steps_total"],
                tool_calls_total=metrics_data["tool_calls_total"],
                tool_calls_successful=metrics_data["tool_calls_successful"],
                tool_calls_failed=metrics_data["tool_calls_failed"],
                invalid_tool_calls=metrics_data["invalid_tool_calls"],
                governance_denials=metrics_data["governance_denials"],
                rag_queries_total=metrics_data.get("rag_queries_total", 0),
                rag_sources_retrieved_total=metrics_data.get(
                    "rag_sources_retrieved_total",
                    0,
                ),
                has_final_answer=metrics_data.get("has_final_answer", False),
                final_answer_length=metrics_data.get("final_answer_length", 0),
                rag_sources_available_count=metrics_data.get(
                    "rag_sources_available_count",
                    0,
                ),
                rag_unique_chunks_count=metrics_data.get(
                    "rag_unique_chunks_count",
                    0,
                ),
                task_completed=metrics_data["task_completed"],
            ),
            policy=AgentEvaluationPolicy(
                max_execution_time_ms=policy_data.get("max_execution_time_ms"),
                max_steps_per_run=policy_data.get("max_steps_per_run"),
                max_invalid_tool_calls=policy_data.get("max_invalid_tool_calls"),
                allow_governance_denials=policy_data.get(
                    "allow_governance_denials",
                    False,
                ),
                require_task_completed=policy_data.get(
                    "require_task_completed",
                    True,
                ),
                require_answer_match=policy_data.get(
                    "require_answer_match",
                    False,
                ),
                name=policy_data.get("name"),
            ),
            quality_gate=AgentQualityGateResult(
                passed=quality_gate_data["passed"],
                violations=tuple(quality_gate_data.get("violations", [])),
            ),
            answer_evaluation=(
                AgentAnswerEvaluation(
                    evaluated=answer_evaluation_data["evaluated"],
                    exact_match=answer_evaluation_data["exact_match"],
                    normalization=answer_evaluation_data.get(
                        "normalization",
                        "whitespace_casefold",
                    ),
                )
                if answer_evaluation_data is not None
                else None
            ),
        )
