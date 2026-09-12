from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from app.control_plane.dependencies import (
    get_external_evaluation_release_policy,
    get_retrieval_evaluation_run_store,
)
from app.control_plane.schemas.evaluation import (
    CompositeEvaluationReleaseDecisionResponse,
    EvaluationComparisonResponse,
    EvaluationReleaseDecisionResponse,
    EvaluationRunListResponse,
    EvaluationRunResponse,
    EvaluationRunSummaryResponse,
)
from rag.evaluation.release import RetrievalEvaluationReleaseGate
from rag.evaluation.run_store import RetrievalEvaluationRunStore
from rag.evaluation.composite_release import CompositeEvaluationReleaseGate
from rag.evaluation.external.release import (
    ExternalEvaluationReleaseGate,
    ExternalEvaluationReleasePolicy,
)

router = APIRouter(
    prefix="/api/v1/evaluation",
    tags=["evaluation"],
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
