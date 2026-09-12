from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.control_plane.persistence.models import RetrievalEvaluationRunRecord
from rag.evaluation.comparison.regression_policy import (
    RetrievalRegressionPolicy,
)
from rag.evaluation.comparison.run_comparator import (
    RetrievalEvaluationMetricComparison,
    RetrievalEvaluationMetricStatus,
    RetrievalEvaluationRunComparison,
)
from rag.evaluation.external import (
    ExternalEvaluationMetricPolicy,
    ExternalEvaluationPolicy,
    ExternalEvaluationQualityGateResult,
    ExternalEvaluationResult,
)
from rag.evaluation.lineage import RetrievalEvaluationLineage
from rag.evaluation.models import RetrievalEvaluationResult
from rag.evaluation.policy import RetrievalEvaluationPolicy
from rag.evaluation.quality_gate import RetrievalQualityGateResult
from rag.evaluation.run import (
    RetrievalEvaluationRegression,
    RetrievalEvaluationRun,
)
from rag.evaluation.run_store import (
    DuplicateEvaluationRunError,
    RetrievalEvaluationRunStore,
)
from rag.models import EmbeddingIdentity


class PostgreSQLRetrievalEvaluationRunStore(RetrievalEvaluationRunStore):
    """
    Durable PostgreSQL persistence for retrieval evaluation runs.

    Only aggregate evaluation evidence is persisted. Per-query evaluation
    payloads, including query text, retrieved chunk IDs, scores, and error
    messages, are intentionally excluded.
    """

    def __init__(self, session: Session) -> None:
        self._session = session

    async def save(
        self,
        run: RetrievalEvaluationRun,
        *,
        commit: bool = True,
    ) -> None:
        try:
            existing = self._session.scalar(
                select(RetrievalEvaluationRunRecord).where(
                    RetrievalEvaluationRunRecord.run_id == run.run_id
                )
            )

            if existing is not None:
                raise DuplicateEvaluationRunError(f"evaluation run already exists: {run.run_id}")

            payload = _serialize_run(run)

            self._session.add(RetrievalEvaluationRunRecord(**payload))

            self._session.flush()

            if commit:
                self._session.commit()

        except DuplicateEvaluationRunError:
            if commit:
                self._session.rollback()
            raise

        except IntegrityError as exc:
            if commit:
                self._session.rollback()

            raise DuplicateEvaluationRunError(
                f"evaluation run already exists: {run.run_id}"
            ) from exc

        except Exception:
            if commit:
                self._session.rollback()
            raise

    async def get(self, run_id: str) -> RetrievalEvaluationRun | None:
        record = self._session.scalar(
            select(RetrievalEvaluationRunRecord).where(
                RetrievalEvaluationRunRecord.run_id == run_id
            )
        )

        if record is None:
            return None

        return _deserialize_run(record)

    async def list(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> list[RetrievalEvaluationRun]:
        records = self._session.scalars(
            select(RetrievalEvaluationRunRecord)
            .order_by(
                RetrievalEvaluationRunRecord.created_at.desc(),
                RetrievalEvaluationRunRecord.run_id.desc(),
            )
            .offset(offset)
            .limit(limit)
        ).all()

        return [_deserialize_run(record) for record in records]

    async def count(self) -> int:
        return int(
            self._session.scalar(select(func.count()).select_from(RetrievalEvaluationRunRecord))
            or 0
        )


def _serialize_run(
    run: RetrievalEvaluationRun,
) -> dict[str, object]:
    lineage = run.lineage.as_dict()

    evaluation = {
        "retrieval_recall_at_k": run.evaluation.recall_at_k,
        "retrieval_precision_at_k": run.evaluation.precision_at_k,
        "retrieval_mrr": run.evaluation.mrr,
        "retrieval_ndcg_at_k": run.evaluation.ndcg_at_k,
        "retrieval_evaluated_queries": run.evaluation.evaluated_queries,
        "retrieval_successful_queries": run.evaluation.successful_queries,
        "retrieval_failed_queries": run.evaluation.failed_queries,
        "retrieval_mean_latency_ms": run.evaluation.mean_latency_ms,
        "retrieval_abstention_accuracy": run.evaluation.abstention_accuracy,
        "retrieval_abstention_evaluated_queries": (run.evaluation.abstention_evaluated_queries),
    }

    quality_gate = run.quality_gate.as_dict()
    quality_gate["quality_gate_policy"] = run.quality_gate.policy.name
    quality_gate["quality_gate_errors"] = list(run.quality_gate.errors)
    quality_gate["quality_gate_metrics"] = dict(run.quality_gate.metrics)

    regression = _serialize_regression(run.regression) if run.regression is not None else None
    external_evaluations = [result.as_dict() for result in run.external_evaluations]
    if run.external_quality_gate is not None:
        external_quality_gate = run.external_quality_gate.as_dict()
        external_quality_gate["quality_gate_policy_data"] = (
            run.external_quality_gate.policy.as_dict()
        )
    else:
        external_quality_gate = None

    return {
        "run_id": run.run_id,
        "created_at": run.created_at,
        "dataset_name": run.lineage.dataset_name,
        "dataset_version": run.lineage.dataset_version,
        "release_passed": run.release_passed,
        "lineage": lineage,
        "evaluation": evaluation,
        "quality_gate": quality_gate,
        "regression": regression,
        "external_evaluations": external_evaluations,
        "external_quality_gate": external_quality_gate,
    }


def _serialize_regression(
    regression: RetrievalEvaluationRegression,
) -> dict[str, object]:
    comparison = regression.comparison

    return {
        "baseline_run_id": regression.baseline_run_id,
        "candidate_run_id": regression.candidate_run_id,
        "comparison": comparison.as_dict(),
        "policy": regression.policy.as_dict(),
        "result": regression.result.as_dict(),
    }


def _deserialize_run(
    record: RetrievalEvaluationRunRecord,
) -> RetrievalEvaluationRun:
    lineage_data = dict(record.lineage)

    embedding_identity = None

    if lineage_data["embedding_requested_model"] is not None:
        embedding_identity = EmbeddingIdentity(
            requested_provider=lineage_data["embedding_requested_provider"],
            requested_model=lineage_data["embedding_requested_model"],
            resolved_provider=lineage_data["embedding_resolved_provider"],
            resolved_model=lineage_data["embedding_resolved_model"],
            dimension=lineage_data["embedding_dimension"],
        )

    lineage = RetrievalEvaluationLineage(
        dataset_name=lineage_data["dataset_name"],
        dataset_version=lineage_data["dataset_version"],
        evaluation_policy_name=lineage_data["evaluation_policy_name"],
        min_recall_at_k=lineage_data["min_recall_at_k"],
        min_precision_at_k=lineage_data["min_precision_at_k"],
        min_mrr=lineage_data["min_mrr"],
        min_ndcg_at_k=lineage_data["min_ndcg_at_k"],
        max_mean_latency_ms=lineage_data["max_mean_latency_ms"],
        min_abstention_accuracy=lineage_data["min_abstention_accuracy"],
        evaluator_k=lineage_data["evaluator_k"],
        min_relevance_score=lineage_data["min_relevance_score"],
        embedding_identity=embedding_identity,
    )

    evaluation_data = dict(record.evaluation)

    evaluation = RetrievalEvaluationResult(
        recall_at_k=evaluation_data["retrieval_recall_at_k"],
        precision_at_k=evaluation_data["retrieval_precision_at_k"],
        mrr=evaluation_data["retrieval_mrr"],
        ndcg_at_k=evaluation_data["retrieval_ndcg_at_k"],
        evaluated_queries=evaluation_data["retrieval_evaluated_queries"],
        successful_queries=evaluation_data["retrieval_successful_queries"],
        failed_queries=evaluation_data["retrieval_failed_queries"],
        mean_latency_ms=evaluation_data["retrieval_mean_latency_ms"],
        query_results=(),
        abstention_accuracy=evaluation_data["retrieval_abstention_accuracy"],
        abstention_evaluated_queries=evaluation_data["retrieval_abstention_evaluated_queries"],
    )

    quality_gate_data = dict(record.quality_gate)
    policy = RetrievalEvaluationPolicy(
        min_recall_at_k=lineage.min_recall_at_k,
        min_precision_at_k=lineage.min_precision_at_k,
        min_mrr=lineage.min_mrr,
        min_ndcg_at_k=lineage.min_ndcg_at_k,
        max_mean_latency_ms=lineage.max_mean_latency_ms,
        min_abstention_accuracy=lineage.min_abstention_accuracy,
        name=quality_gate_data["quality_gate_policy"],
    )

    quality_gate = RetrievalQualityGateResult(
        passed=quality_gate_data["quality_gate_passed"],
        errors=tuple(quality_gate_data["quality_gate_errors"]),
        metrics=dict(quality_gate_data["quality_gate_metrics"]),
        policy=policy,
    )

    external_evaluations = tuple(
        ExternalEvaluationResult(
            provider=data["provider"],
            evaluator=data["evaluator"],
            metrics=data["metrics"],
            evaluated_samples=data["evaluated_samples"],
            metadata=data.get("metadata"),
        )
        for data in (record.external_evaluations or [])
    )

    external_quality_gate = _deserialize_external_quality_gate(record.external_quality_gate)

    run = RetrievalEvaluationRun(
        run_id=record.run_id,
        created_at=_ensure_aware(record.created_at),
        lineage=lineage,
        evaluation=evaluation,
        quality_gate=quality_gate,
        external_evaluations=external_evaluations,
        external_quality_gate=external_quality_gate,
    )

    if record.regression is None:
        return run

    regression = _deserialize_regression(record.regression)

    return RetrievalEvaluationRun(
        run_id=run.run_id,
        created_at=run.created_at,
        lineage=run.lineage,
        evaluation=run.evaluation,
        quality_gate=run.quality_gate,
        regression=regression,
        external_evaluations=run.external_evaluations,
        external_quality_gate=run.external_quality_gate,
    )


def _deserialize_external_quality_gate(
    data: dict | None,
) -> ExternalEvaluationQualityGateResult | None:
    if data is None:
        return None

    policy_data = data["quality_gate_policy_data"]
    policy = ExternalEvaluationPolicy(
        name=policy_data["name"],
        metrics=tuple(
            ExternalEvaluationMetricPolicy(
                provider=metric["provider"],
                evaluator=metric["evaluator"],
                metric_name=metric["metric_name"],
                minimum_value=metric["minimum_value"],
                maximum_value=metric["maximum_value"],
                required=metric["required"],
            )
            for metric in policy_data["metrics"]
        ),
    )

    return ExternalEvaluationQualityGateResult(
        passed=data["quality_gate_passed"],
        errors=tuple(data["quality_gate_errors"]),
        metrics=dict(data["quality_gate_metrics"]),
        policy=policy,
    )


def _deserialize_regression(
    data: dict,
) -> RetrievalEvaluationRegression:
    comparison_data = data["comparison"]

    metrics = {}

    for name, metric_data in comparison_data["metrics"].items():
        metrics[name] = RetrievalEvaluationMetricComparison(
            baseline=metric_data["baseline"],
            candidate=metric_data["candidate"],
            delta=metric_data["delta"],
            status=RetrievalEvaluationMetricStatus(metric_data["status"]),
        )

    comparison = RetrievalEvaluationRunComparison(
        baseline_run_id=comparison_data["baseline_run_id"],
        candidate_run_id=comparison_data["candidate_run_id"],
        metrics=metrics,
    )

    policy_data = data["policy"]

    policy = RetrievalRegressionPolicy(
        max_recall_at_k_degradation=policy_data["max_recall_at_k_degradation"],
        max_precision_at_k_degradation=policy_data["max_precision_at_k_degradation"],
        max_mrr_degradation=policy_data["max_mrr_degradation"],
        max_ndcg_at_k_degradation=policy_data["max_ndcg_at_k_degradation"],
        max_mean_latency_ms_increase=policy_data["max_mean_latency_ms_increase"],
        max_abstention_accuracy_degradation=policy_data["max_abstention_accuracy_degradation"],
        name=policy_data["name"],
    )

    result_data = data["result"]

    from rag.evaluation.comparison.regression_policy import (
        RetrievalRegressionPolicyResult,
    )

    result = RetrievalRegressionPolicyResult(
        passed=result_data["passed"],
        errors=tuple(result_data["errors"]),
        policy_name=result_data["policy_name"],
    )

    return RetrievalEvaluationRegression(
        baseline_run_id=data["baseline_run_id"],
        candidate_run_id=data["candidate_run_id"],
        comparison=comparison,
        policy=policy,
        result=result,
    )


def _ensure_aware(value: datetime) -> datetime:
    if value.tzinfo is None:
        # SQLite does not preserve timezone information for DateTime(timezone=True).
        # PostgreSQL returns an aware datetime for the production schema.
        return value.replace(tzinfo=UTC)

    return value
