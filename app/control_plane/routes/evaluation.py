from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, Query, status
from app.control_plane.dependencies import (
    get_evaluation_application_service,
    get_external_evaluation_release_policy,
    get_retrieval_evaluation_release_decision_store,
    get_retrieval_evaluation_run_store,
)
from app.control_plane.schemas.evaluation import (
    CompositeEvaluationReleaseDecisionResponse,
    EvaluationComparisonResponse,
    EvaluationReleaseDecisionResponse,
    EvaluationRunExecutionResponse,
    EvaluationRunListResponse,
    EvaluationRunRequest,
    EvaluationRunResponse,
    EvaluationRunSummaryResponse,
)
from rag.evaluation.release import RetrievalEvaluationReleaseGate
from rag.evaluation.run_store import (
    DuplicateEvaluationRunError,
    RetrievalEvaluationRunStore,
)
from rag.evaluation.release_decision_store import (
    RetrievalEvaluationReleaseDecisionStore,
)
from rag.evaluation.composite_release import CompositeEvaluationReleaseGate
from rag.evaluation.comparison.regression_policy import RetrievalRegressionPolicy
from rag.evaluation.policy import RetrievalEvaluationPolicy
from rag.evaluation.external.release import (
    ExternalEvaluationReleaseGate,
    ExternalEvaluationReleasePolicy,
)
from app.control_plane.evaluation_application_service import (
    EvaluationApplicationService,
    EvaluationBaselineNotFoundError,
)

router = APIRouter(
    prefix="/api/v1/evaluation",
    tags=["evaluation"],
)


@router.post(
    "/runs",
    response_model=EvaluationRunExecutionResponse,
    status_code=status.HTTP_201_CREATED,
)
async def execute_evaluation_run(
    request: EvaluationRunRequest,
    service: EvaluationApplicationService = Depends(get_evaluation_application_service),
    external_release_policy: ExternalEvaluationReleasePolicy = Depends(
        get_external_evaluation_release_policy
    ),
) -> EvaluationRunExecutionResponse:
    try:
        evaluation_policy = RetrievalEvaluationPolicy(
            name=request.evaluation_policy.name,
            min_recall_at_k=request.evaluation_policy.min_recall_at_k,
            min_precision_at_k=request.evaluation_policy.min_precision_at_k,
            min_mrr=request.evaluation_policy.min_mrr,
            min_ndcg_at_k=request.evaluation_policy.min_ndcg_at_k,
            max_mean_latency_ms=request.evaluation_policy.max_mean_latency_ms,
            min_abstention_accuracy=(request.evaluation_policy.min_abstention_accuracy),
        )

        regression_policy = None

        if request.regression_policy is not None:
            regression_policy = RetrievalRegressionPolicy(
                name=request.regression_policy.name,
                max_recall_at_k_degradation=(request.regression_policy.max_recall_at_k_degradation),
                max_precision_at_k_degradation=(
                    request.regression_policy.max_precision_at_k_degradation
                ),
                max_mrr_degradation=(request.regression_policy.max_mrr_degradation),
                max_ndcg_at_k_degradation=(request.regression_policy.max_ndcg_at_k_degradation),
                max_mean_latency_ms_increase=(
                    request.regression_policy.max_mean_latency_ms_increase
                ),
                max_abstention_accuracy_degradation=(
                    request.regression_policy.max_abstention_accuracy_degradation
                ),
            )

        result = await service.execute(
            dataset_name=request.dataset_name,
            dataset_version=request.dataset_version,
            run_id=request.run_id,
            created_at=datetime.now(UTC),
            evaluation_policy=evaluation_policy,
            external_release_policy=external_release_policy,
            k=request.k,
            min_relevance_score=request.min_relevance_score,
            baseline_run_id=request.baseline_run_id,
            regression_policy=regression_policy,
        )

    except DuplicateEvaluationRunError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc
    except EvaluationBaselineNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from exc

    run = result.run
    decision = result.release_decision

    return EvaluationRunExecutionResponse(
        run=EvaluationRunResponse(
            run_id=run.run_id,
            created_at=run.created_at,
            dataset_name=run.lineage.dataset_name,
            dataset_version=run.lineage.dataset_version,
            lineage=run.lineage.as_dict(),
            evaluation=run.evaluation.as_dict(),
            quality_gate=run.quality_gate.as_dict(),
            regression=(run.regression.as_dict() if run.regression is not None else None),
            external_evaluations=[evaluation.as_dict() for evaluation in run.external_evaluations],
            external_quality_gate=(
                run.external_quality_gate.as_dict()
                if run.external_quality_gate is not None
                else None
            ),
            passed=run.passed,
            release_passed=run.release_passed,
        ),
        release_decision=CompositeEvaluationReleaseDecisionResponse(
            run_id=decision.run_id,
            passed=decision.passed,
            errors=list(decision.errors),
            native=decision.native.as_dict(),
            external=decision.external.as_dict(),
        ),
    )


@router.get(
    "/runs",
    response_model=EvaluationRunListResponse,
)
async def list_evaluation_runs(
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    store: RetrievalEvaluationRunStore = Depends(get_retrieval_evaluation_run_store),
) -> EvaluationRunListResponse:
    runs = await store.list(
        limit=limit,
        offset=offset,
    )
    total = await store.count()

    return EvaluationRunListResponse(
        runs=[
            EvaluationRunSummaryResponse(
                run_id=run.run_id,
                created_at=run.created_at,
                dataset_name=run.lineage.dataset_name,
                dataset_version=run.lineage.dataset_version,
                release_passed=run.release_passed,
            )
            for run in runs
        ],
        limit=limit,
        offset=offset,
        total=total,
    )


@router.get(
    "/runs/{run_id}",
    response_model=EvaluationRunResponse,
)
async def get_evaluation_run(
    run_id: str,
    store: RetrievalEvaluationRunStore = Depends(get_retrieval_evaluation_run_store),
) -> EvaluationRunResponse:
    run = await store.get(run_id)

    if run is None:
        raise HTTPException(
            status_code=404,
            detail=f"evaluation run not found: {run_id}",
        )

    return EvaluationRunResponse(
        run_id=run.run_id,
        created_at=run.created_at,
        dataset_name=run.lineage.dataset_name,
        dataset_version=run.lineage.dataset_version,
        lineage=run.lineage.as_dict(),
        evaluation=run.evaluation.as_dict(),
        quality_gate=run.quality_gate.as_dict(),
        regression=(run.regression.as_dict() if run.regression is not None else None),
        external_evaluations=[result.as_dict() for result in run.external_evaluations],
        external_quality_gate=(
            run.external_quality_gate.as_dict() if run.external_quality_gate is not None else None
        ),
        passed=run.passed,
        release_passed=run.release_passed,
    )


@router.get(
    "/runs/{run_id}/comparison",
    response_model=EvaluationComparisonResponse,
)
async def get_evaluation_comparison(
    run_id: str,
    store: RetrievalEvaluationRunStore = Depends(get_retrieval_evaluation_run_store),
) -> EvaluationComparisonResponse:
    run = await store.get(run_id)

    if run is None:
        raise HTTPException(
            status_code=404,
            detail=f"evaluation run not found: {run_id}",
        )

    if run.regression is None:
        raise HTTPException(
            status_code=404,
            detail=f"evaluation run has no regression comparison: {run_id}",
        )

    comparison = run.regression.comparison

    return EvaluationComparisonResponse(
        baseline_run_id=comparison.baseline_run_id,
        candidate_run_id=comparison.candidate_run_id,
        metrics=comparison.as_dict()["metrics"],
    )


@router.get(
    "/runs/{run_id}/release-decision",
    response_model=EvaluationReleaseDecisionResponse,
)
async def get_evaluation_release_decision(
    run_id: str,
    store: RetrievalEvaluationRunStore = Depends(get_retrieval_evaluation_run_store),
) -> EvaluationReleaseDecisionResponse:
    run = await store.get(run_id)

    if run is None:
        raise HTTPException(
            status_code=404,
            detail=f"evaluation run not found: {run_id}",
        )

    decision = RetrievalEvaluationReleaseGate.evaluate(run)

    return EvaluationReleaseDecisionResponse(
        run_id=decision.run_id,
        passed=decision.passed,
        errors=list(decision.errors),
    )


@router.get(
    "/runs/{run_id}/composite-release-decision",
    response_model=CompositeEvaluationReleaseDecisionResponse,
)
async def get_composite_evaluation_release_decision(
    run_id: str,
    store: RetrievalEvaluationRunStore = Depends(get_retrieval_evaluation_run_store),
    decision_store: RetrievalEvaluationReleaseDecisionStore = Depends(
        get_retrieval_evaluation_release_decision_store
    ),
    external_policy: ExternalEvaluationReleasePolicy = Depends(
        get_external_evaluation_release_policy
    ),
) -> CompositeEvaluationReleaseDecisionResponse:
    run = await store.get(run_id)

    if run is None:
        raise HTTPException(
            status_code=404,
            detail=f"evaluation run not found: {run_id}",
        )

    persisted_decision = await decision_store.get(run_id)

    if persisted_decision is not None:
        decision = persisted_decision
    else:
        native = RetrievalEvaluationReleaseGate.evaluate(run)

        external = ExternalEvaluationReleaseGate.evaluate(
            quality_gate=run.external_quality_gate,
            policy=external_policy,
        )

        decision = CompositeEvaluationReleaseGate.evaluate(
            run=run,
            native=native,
            external=external,
        )

    return CompositeEvaluationReleaseDecisionResponse(
        run_id=decision.run_id,
        passed=decision.passed,
        errors=list(decision.errors),
        native=decision.native.as_dict(),
        external=decision.external.as_dict(),
    )
