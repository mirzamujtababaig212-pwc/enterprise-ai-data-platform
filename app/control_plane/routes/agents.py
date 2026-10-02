from __future__ import annotations
from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request, status

from ai_platform.agents.observability import AgentExecutionEventType
from ai_platform.agents.evaluation.policy import AgentEvaluationPolicy
from ai_platform.agents.models import AgentRequest

from app.control_plane.agent_evaluations.application_service import (
    AgentEvaluationApplicationService,
)
from app.control_plane.agent_evaluations.repository import (
    DuplicateAgentEvaluationRunError,
)
from app.control_plane.agent_runs.application_service import (
    AgentRunApplicationService,
)
from app.control_plane.agent_runs.exceptions import (
    AgentRunAccessDeniedError,
    AgentRunIdempotencyConflictError,
)
from app.control_plane.agent_run_steps.models import AgentRunStepStatus
from app.control_plane.agent_runs.models import AgentRunStatus
from app.control_plane.dependencies import (
    get_agent_evaluation_application_service,
    get_agent_run_application_service,
    get_agent_run_recovery_service,
)
from app.control_plane.schemas.agent_evaluation import (
    AgentAnswerEvaluationResponse,
    AgentEvaluationLineageResponse,
    AgentEvaluationMetricsResponse,
    AgentEvaluationDiagnosticsResponse,
    AgentEvaluationContextQualityResponse,
    AgentGroundingClaimAttributionResponse,
    AgentEvaluationPolicyRequest,
    AgentEvaluationRequest,
    AgentEvaluationQualityGateResponse,
    AgentEvaluationRunListResponse,
    AgentEvaluationRunResponse,
)
from app.control_plane.schemas.agents import (
    AgentRunCancellationResponse,
    AgentRunDetailResponse,
    AgentRunEventListResponse,
    AgentRunEventResponse,
    AgentRunListResponse,
    AgentRunRequest,
    AgentRunResponse,
    AgentRunStepListResponse,
    AgentRunStepResponse,
)
from app.control_plane.agent_runs.recovery_service import (
    AgentRunRecoveryService,
)

router = APIRouter(
    prefix="/api/v1/agents",
    tags=["agents"],
)


@router.post(
    "/{agent_name}/run",
    response_model=AgentRunResponse,
)
async def run_agent(
    request: Request,
    agent_name: str,
    payload: AgentRunRequest,
    idempotency_key: str | None = Header(
        default=None,
        alias="Idempotency-Key",
    ),
    service: AgentRunApplicationService = Depends(
        get_agent_run_application_service,
    ),
) -> AgentRunResponse:
    try:
        agent_request = AgentRequest(
            input=payload.input,
            session_id=payload.session_id,
            user_id=payload.user_id,
            principal=getattr(request.state, "principal", None),
            tenant_id=getattr(request.state, "tenant_id", None),
            metadata=payload.metadata,
        )

        response = await service.execute(
            agent_name=agent_name,
            request=agent_request,
            idempotency_key=idempotency_key,
        )

    except LookupError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    except AgentRunIdempotencyConflictError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from exc

    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc

    return AgentRunResponse(
        run_id=response.run_id,
        agent_name=response.response.agent_name,
        output=response.response.output,
        session_id=response.response.session_id,
        metadata=response.response.metadata,
    )


@router.post(
    "/runs/{run_id}/recover",
    response_model=AgentRunResponse,
)
async def recover_agent_run(
    request: Request,
    run_id: str,
    application_service: AgentRunApplicationService = Depends(
        get_agent_run_application_service,
    ),
    service: AgentRunRecoveryService = Depends(
        get_agent_run_recovery_service,
    ),
) -> AgentRunResponse:
    try:
        tenant_id = getattr(request.state, "tenant_id", None)
        principal = getattr(request.state, "principal", None)

        if tenant_id is None or principal is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent run '{run_id}' was not found.",
            )

        run = application_service.get_run(
            run_id,
            tenant_id=tenant_id,
            principal=principal,
        )

        if run is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent run '{run_id}' was not found.",
            )

        response = await service.recover(run_id)

    except AgentRunAccessDeniedError as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(exc),
        ) from exc

    except LookupError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from exc

    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc

    return AgentRunResponse(
        run_id=response.run_id,
        agent_name=response.response.agent_name,
        output=response.response.output,
        session_id=response.response.session_id,
        metadata=response.response.metadata,
    )


@router.get(
    "/runs",
    response_model=AgentRunListResponse,
)
async def list_agent_runs(
    request: Request,
    agent_name: str | None = Query(default=None),
    session_id: str | None = Query(default=None),
    user_id: str | None = Query(default=None),
    run_status: str | None = Query(default=None, alias="status"),
    limit: int = Query(default=100, ge=1, le=100),
    service: AgentRunApplicationService = Depends(
        get_agent_run_application_service,
    ),
) -> AgentRunListResponse:
    selected_status = None

    if run_status is not None:
        try:
            selected_status = AgentRunStatus(run_status)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=f"Invalid agent run status: {run_status}",
            ) from exc

    tenant_id = getattr(request.state, "tenant_id", None)

    if tenant_id is None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Authenticated tenant context is required.",
        )

    runs = service.list_runs(
        tenant_id=tenant_id,
        agent_name=agent_name,
        session_id=session_id,
        user_id=user_id,
        status=selected_status,
        limit=limit,
    )

    return AgentRunListResponse(
        runs=[
            AgentRunDetailResponse(
                run_id=run.run_id,
                agent_name=run.agent_name,
                status=run.status.value,
                session_id=run.session_id,
                started_at=run.started_at,
                completed_at=run.completed_at,
                output=run.output,
                metadata=run.metadata,
            )
            for run in runs
        ]
    )


@router.post(
    "/runs/{run_id}/cancel",
    response_model=AgentRunCancellationResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def cancel_agent_run(
    request: Request,
    run_id: str,
    service: AgentRunApplicationService = Depends(
        get_agent_run_application_service,
    ),
) -> AgentRunCancellationResponse:
    try:
        run = service.cancel(
            run_id,
            tenant_id=getattr(request.state, "tenant_id", None),
            principal=getattr(request.state, "principal", None),
        )

    except AgentRunAccessDeniedError as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(exc),
        ) from exc

    except LookupError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc

    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc

    return AgentRunCancellationResponse(
        run_id=run.run_id,
        status=run.status.value,
        message="Cancellation requested.",
    )


@router.get(
    "/runs/{run_id}",
    response_model=AgentRunDetailResponse,
)
async def get_agent_run(
    request: Request,
    run_id: str,
    service: AgentRunApplicationService = Depends(
        get_agent_run_application_service,
    ),
) -> AgentRunDetailResponse:
    try:
        run = service.get_run(
            run_id,
            tenant_id=getattr(request.state, "tenant_id", None),
            principal=getattr(request.state, "principal", None),
        )
    except AgentRunAccessDeniedError as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(exc),
        ) from exc

    if run is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent run '{run_id}' was not found.",
        )

    return AgentRunDetailResponse(
        run_id=run.run_id,
        agent_name=run.agent_name,
        status=run.status.value,
        session_id=run.session_id,
        started_at=run.started_at,
        completed_at=run.completed_at,
        output=run.output,
        metadata=run.metadata,
    )


@router.get(
    "/runs/{run_id}/steps",
    response_model=AgentRunStepListResponse,
)
async def list_agent_run_steps(
    request: Request,
    run_id: str,
    run_status: str | None = Query(default=None, alias="status"),
    limit: int = Query(default=100, ge=1, le=100),
    service: AgentRunApplicationService = Depends(
        get_agent_run_application_service,
    ),
) -> AgentRunStepListResponse:
    selected_status = None

    if run_status is not None:
        try:
            selected_status = AgentRunStepStatus(run_status)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=f"Invalid agent run step status: {run_status}",
            ) from exc

    try:
        steps = service.list_steps(
            run_id,
            tenant_id=getattr(request.state, "tenant_id", None),
            principal=getattr(request.state, "principal", None),
            status=selected_status,
            limit=limit,
        )
    except AgentRunAccessDeniedError as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(exc),
        ) from exc
    except LookupError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    return AgentRunStepListResponse(
        steps=[
            AgentRunStepResponse(
                run_id=step.run_id,
                step_id=step.step_id,
                step_index=step.step_index,
                step_type=step.step_type,
                status=step.status.value,
                attempt=step.attempt,
                tool_name=step.tool_name,
                call_id=step.call_id,
                input=step.input,
                output=step.output,
                error=step.error,
                failure_category=step.failure_category,
                started_at=step.started_at,
                completed_at=step.completed_at,
                created_at=step.created_at,
                updated_at=step.updated_at,
                metadata=step.metadata,
            )
            for step in steps
        ]
    )


@router.get(
    "/runs/{run_id}/steps/{step_id}",
    response_model=AgentRunStepResponse,
)
async def get_agent_run_step(
    request: Request,
    run_id: str,
    step_id: str,
    service: AgentRunApplicationService = Depends(
        get_agent_run_application_service,
    ),
) -> AgentRunStepResponse:
    try:
        step = service.get_step(
            run_id,
            step_id,
            tenant_id=getattr(request.state, "tenant_id", None),
            principal=getattr(request.state, "principal", None),
        )
    except AgentRunAccessDeniedError as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(exc),
        ) from exc
    except LookupError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    if step is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(f"Agent run step '{step_id}' for run " f"'{run_id}' was not found."),
        )

    return AgentRunStepResponse(
        run_id=step.run_id,
        step_id=step.step_id,
        step_index=step.step_index,
        step_type=step.step_type,
        status=step.status.value,
        attempt=step.attempt,
        tool_name=step.tool_name,
        call_id=step.call_id,
        input=step.input,
        output=step.output,
        error=step.error,
        failure_category=step.failure_category,
        started_at=step.started_at,
        completed_at=step.completed_at,
        created_at=step.created_at,
        updated_at=step.updated_at,
        metadata=step.metadata,
    )


@router.post(
    "/runs/{run_id}/evaluations",
    response_model=AgentEvaluationRunResponse,
)
async def evaluate_agent_run(
    request: Request,
    run_id: str,
    payload: AgentEvaluationRequest,
    service: AgentEvaluationApplicationService = Depends(
        get_agent_evaluation_application_service,
    ),
) -> AgentEvaluationRunResponse:
    tenant_id = getattr(request.state, "tenant_id", None)
    principal = getattr(request.state, "principal", None)

    if tenant_id is None or principal is None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Both tenant_id and principal are required for evaluation.",
        )

    policy = AgentEvaluationPolicy(
        policy_id=payload.policy_id,
        policy_version=payload.policy_version,
        max_execution_time_ms=payload.max_execution_time_ms,
        max_steps_per_run=payload.max_steps_per_run,
        max_invalid_tool_calls=payload.max_invalid_tool_calls,
        min_retrieval_score=payload.min_retrieval_score,
        min_reranker_score=payload.min_reranker_score,
        min_grounding_support_ratio=payload.min_grounding_support_ratio,
        min_semantic_grounding_score=payload.min_semantic_grounding_score,
        allow_governance_denials=payload.allow_governance_denials,
        require_task_completed=payload.require_task_completed,
        require_answer_match=payload.require_answer_match,
        require_rag_provenance=payload.require_rag_provenance,
        name=payload.name,
    )

    try:
        evaluation = await service.evaluate_run(
            run_id,
            tenant_id=tenant_id,
            principal=principal,
            policy=policy,
            expected_answer=payload.expected_answer,
        )
    except PermissionError as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(exc),
        ) from exc
    except LookupError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except DuplicateAgentEvaluationRunError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from exc

    return AgentEvaluationRunResponse(
        evaluation_run_id=evaluation.evaluation_run_id,
        created_at=evaluation.created_at,
        passed=evaluation.passed,
        lineage=AgentEvaluationLineageResponse(
            evaluated_run_id=evaluation.lineage.evaluated_run_id,
            agent_name=evaluation.lineage.agent_name,
            agent_version=evaluation.lineage.agent_version,
            tenant_id=evaluation.lineage.tenant_id,
            effective_model=evaluation.lineage.effective_model,
            effective_provider=evaluation.lineage.effective_provider,
            model_policy_id=evaluation.lineage.model_policy_id,
            model_policy_version=evaluation.lineage.model_policy_version,
        ),
        metrics=AgentEvaluationMetricsResponse(
            execution_time_ms=evaluation.metrics.execution_time_ms,
            steps_total=evaluation.metrics.steps_total,
            tool_calls_total=evaluation.metrics.tool_calls_total,
            tool_calls_successful=evaluation.metrics.tool_calls_successful,
            tool_calls_failed=evaluation.metrics.tool_calls_failed,
            invalid_tool_calls=evaluation.metrics.invalid_tool_calls,
            governance_denials=evaluation.metrics.governance_denials,
            rag_queries_total=evaluation.metrics.rag_queries_total,
            rag_sources_retrieved_total=evaluation.metrics.rag_sources_retrieved_total,
            has_final_answer=evaluation.metrics.has_final_answer,
            final_answer_length=evaluation.metrics.final_answer_length,
            has_rag_provenance=evaluation.metrics.has_rag_provenance,
            has_rag_sources_available=evaluation.metrics.has_rag_sources_available,
            rag_sources_available_count=evaluation.metrics.rag_sources_available_count,
            rag_unique_chunks_count=evaluation.metrics.rag_unique_chunks_count,
            grounding_evaluated=evaluation.metrics.grounding_evaluated,
            grounding_supported=evaluation.metrics.grounding_supported,
            grounding_support_ratio=evaluation.metrics.grounding_support_ratio,
            grounding_supported_sources_total=(
                evaluation.metrics.grounding_supported_sources_total
            ),
            grounding_source_candidates_total=(
                evaluation.metrics.grounding_source_candidates_total
            ),
            grounding_method=evaluation.metrics.grounding_method,
            semantic_grounding_evaluated=evaluation.metrics.semantic_grounding_evaluated,
            semantic_grounding_score=evaluation.metrics.semantic_grounding_score,
            semantic_grounding_passed=evaluation.metrics.semantic_grounding_passed,
            semantic_grounding_method=evaluation.metrics.semantic_grounding_method,
            semantic_grounding_evaluator_model=(
                evaluation.metrics.semantic_grounding_evaluator_model
            ),
            semantic_grounding_evaluator_provider=(
                evaluation.metrics.semantic_grounding_evaluator_provider
            ),
            task_completed=evaluation.metrics.task_completed,
        ),
        policy=AgentEvaluationPolicyRequest(
            policy_id=evaluation.policy.policy_id,
            policy_version=evaluation.policy.policy_version,
            max_execution_time_ms=evaluation.policy.max_execution_time_ms,
            max_steps_per_run=evaluation.policy.max_steps_per_run,
            max_invalid_tool_calls=evaluation.policy.max_invalid_tool_calls,
            min_retrieval_score=evaluation.policy.min_retrieval_score,
            min_reranker_score=evaluation.policy.min_reranker_score,
            min_grounding_support_ratio=evaluation.policy.min_grounding_support_ratio,
            min_semantic_grounding_score=evaluation.policy.min_semantic_grounding_score,
            allow_governance_denials=evaluation.policy.allow_governance_denials,
            require_task_completed=evaluation.policy.require_task_completed,
            require_answer_match=evaluation.policy.require_answer_match,
            require_rag_provenance=evaluation.policy.require_rag_provenance,
            name=evaluation.policy.name,
        ),
        quality_gate=AgentEvaluationQualityGateResponse(
            passed=evaluation.quality_gate.passed,
            violations=list(evaluation.quality_gate.violations),
        ),
        answer_evaluation=(
            AgentAnswerEvaluationResponse(
                evaluated=evaluation.answer_evaluation.evaluated,
                exact_match=evaluation.answer_evaluation.exact_match,
                normalization=evaluation.answer_evaluation.normalization,
                semantic_evaluated=evaluation.answer_evaluation.semantic_evaluated,
                semantic_score=evaluation.answer_evaluation.semantic_score,
                semantic_passed=evaluation.answer_evaluation.semantic_passed,
                semantic_method=evaluation.answer_evaluation.semantic_method,
                evaluator_model=evaluation.answer_evaluation.evaluator_model,
                evaluator_provider=evaluation.answer_evaluation.evaluator_provider,
            )
            if evaluation.answer_evaluation is not None
            else None
        ),
        context_quality=(
            AgentEvaluationContextQualityResponse(
                budget_compliant=evaluation.context_quality.budget_compliant,
                retrieval_evidence_present=(evaluation.context_quality.retrieval_evidence_present),
                has_semantic_memory_sources=(
                    evaluation.context_quality.has_semantic_memory_sources
                ),
                has_episodic_memory_sources=(
                    evaluation.context_quality.has_episodic_memory_sources
                ),
                has_working_memory_sources=(evaluation.context_quality.has_working_memory_sources),
                has_chat_history_sources=(evaluation.context_quality.has_chat_history_sources),
                has_tool_result_sources=(evaluation.context_quality.has_tool_result_sources),
                context_source_profile_changes=(
                    evaluation.context_quality.context_source_profile_changes
                ),
                assemblies_total=evaluation.context_quality.assemblies_total,
                messages_total=evaluation.context_quality.messages_total,
                estimated_tokens_total=(evaluation.context_quality.estimated_tokens_total),
                estimated_tokens_max=evaluation.context_quality.estimated_tokens_max,
                minimum_estimated_remaining_after_context=(
                    evaluation.context_quality.minimum_estimated_remaining_after_context
                ),
                source_counts=dict(evaluation.context_quality.source_counts),
            )
            if evaluation.context_quality is not None
            else None
        ),
    )


@router.get(
    "/runs/{run_id}/evaluations/diagnostics",
    response_model=AgentEvaluationDiagnosticsResponse,
)
async def get_agent_run_evaluation_diagnostics(
    request: Request,
    run_id: str,
    service: AgentEvaluationApplicationService = Depends(
        get_agent_evaluation_application_service,
    ),
) -> AgentEvaluationDiagnosticsResponse:
    tenant_id = getattr(request.state, "tenant_id", None)
    principal = getattr(request.state, "principal", None)

    if tenant_id is None or principal is None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Both tenant_id and principal are required for evaluation.",
        )

    try:
        diagnostics = service.get_run_diagnostics(
            run_id,
            tenant_id=tenant_id,
            principal=principal,
        )
    except PermissionError as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(exc),
        ) from exc
    except LookupError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from exc

    return AgentEvaluationDiagnosticsResponse(
        grounding_attributions=[
            AgentGroundingClaimAttributionResponse(
                claim_index=attribution.claim_index,
                claim_text=attribution.claim_text,
                supported=attribution.supported,
                supporting_source_indexes=list(attribution.supporting_source_indexes),
                supporting_source_ids=list(attribution.supporting_source_ids),
            )
            for attribution in diagnostics.grounding_attributions
        ],
    )


@router.get(
    "/runs/{run_id}/evaluations",
    response_model=AgentEvaluationRunListResponse,
)
async def list_agent_run_evaluations(
    request: Request,
    run_id: str,
    limit: int = Query(default=100, ge=1, le=100),
    service: AgentEvaluationApplicationService = Depends(
        get_agent_evaluation_application_service,
    ),
) -> AgentEvaluationRunListResponse:
    tenant_id = getattr(request.state, "tenant_id", None)
    principal = getattr(request.state, "principal", None)

    if tenant_id is None or principal is None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Both tenant_id and principal are required for evaluation.",
        )

    try:
        evaluations = service.list_evaluations(
            run_id,
            tenant_id=tenant_id,
            principal=principal,
            limit=limit,
        )
    except PermissionError as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(exc),
        ) from exc
    except LookupError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from exc

    return AgentEvaluationRunListResponse(
        evaluations=[
            AgentEvaluationRunResponse(
                evaluation_run_id=evaluation.evaluation_run_id,
                created_at=evaluation.created_at,
                passed=evaluation.passed,
                lineage=AgentEvaluationLineageResponse(
                    evaluated_run_id=evaluation.lineage.evaluated_run_id,
                    agent_name=evaluation.lineage.agent_name,
                    agent_version=evaluation.lineage.agent_version,
                    tenant_id=evaluation.lineage.tenant_id,
                    effective_model=evaluation.lineage.effective_model,
                    effective_provider=evaluation.lineage.effective_provider,
                    model_policy_id=evaluation.lineage.model_policy_id,
                    model_policy_version=evaluation.lineage.model_policy_version,
                ),
                metrics=AgentEvaluationMetricsResponse(
                    execution_time_ms=evaluation.metrics.execution_time_ms,
                    steps_total=evaluation.metrics.steps_total,
                    tool_calls_total=evaluation.metrics.tool_calls_total,
                    tool_calls_successful=evaluation.metrics.tool_calls_successful,
                    tool_calls_failed=evaluation.metrics.tool_calls_failed,
                    invalid_tool_calls=evaluation.metrics.invalid_tool_calls,
                    governance_denials=evaluation.metrics.governance_denials,
                    rag_queries_total=evaluation.metrics.rag_queries_total,
                    rag_sources_retrieved_total=evaluation.metrics.rag_sources_retrieved_total,
                    has_final_answer=evaluation.metrics.has_final_answer,
                    has_rag_provenance=evaluation.metrics.has_rag_provenance,
                    has_rag_sources_available=evaluation.metrics.has_rag_sources_available,
                    final_answer_length=evaluation.metrics.final_answer_length,
                    rag_sources_available_count=evaluation.metrics.rag_sources_available_count,
                    rag_unique_chunks_count=evaluation.metrics.rag_unique_chunks_count,
                    grounding_evaluated=evaluation.metrics.grounding_evaluated,
                    grounding_supported=evaluation.metrics.grounding_supported,
                    grounding_support_ratio=evaluation.metrics.grounding_support_ratio,
                    grounding_supported_sources_total=(
                        evaluation.metrics.grounding_supported_sources_total
                    ),
                    grounding_source_candidates_total=(
                        evaluation.metrics.grounding_source_candidates_total
                    ),
                    grounding_method=evaluation.metrics.grounding_method,
                    semantic_grounding_evaluated=evaluation.metrics.semantic_grounding_evaluated,
                    semantic_grounding_score=evaluation.metrics.semantic_grounding_score,
                    semantic_grounding_passed=evaluation.metrics.semantic_grounding_passed,
                    semantic_grounding_method=evaluation.metrics.semantic_grounding_method,
                    semantic_grounding_evaluator_model=(
                        evaluation.metrics.semantic_grounding_evaluator_model
                    ),
                    semantic_grounding_evaluator_provider=(
                        evaluation.metrics.semantic_grounding_evaluator_provider
                    ),
                    task_completed=evaluation.metrics.task_completed,
                ),
                policy=AgentEvaluationPolicyRequest(
                    policy_id=evaluation.policy.policy_id,
                    policy_version=evaluation.policy.policy_version,
                    max_execution_time_ms=evaluation.policy.max_execution_time_ms,
                    max_steps_per_run=evaluation.policy.max_steps_per_run,
                    max_invalid_tool_calls=evaluation.policy.max_invalid_tool_calls,
                    min_retrieval_score=evaluation.policy.min_retrieval_score,
                    min_reranker_score=evaluation.policy.min_reranker_score,
                    min_grounding_support_ratio=evaluation.policy.min_grounding_support_ratio,
                    allow_governance_denials=evaluation.policy.allow_governance_denials,
                    require_task_completed=evaluation.policy.require_task_completed,
                    require_answer_match=evaluation.policy.require_answer_match,
                    require_rag_provenance=evaluation.policy.require_rag_provenance,
                    name=evaluation.policy.name,
                ),
                quality_gate=AgentEvaluationQualityGateResponse(
                    passed=evaluation.quality_gate.passed,
                    violations=list(evaluation.quality_gate.violations),
                ),
                answer_evaluation=(
                    AgentAnswerEvaluationResponse(
                        evaluated=evaluation.answer_evaluation.evaluated,
                        exact_match=evaluation.answer_evaluation.exact_match,
                        normalization=evaluation.answer_evaluation.normalization,
                        semantic_evaluated=evaluation.answer_evaluation.semantic_evaluated,
                        semantic_score=evaluation.answer_evaluation.semantic_score,
                        semantic_passed=evaluation.answer_evaluation.semantic_passed,
                        semantic_method=evaluation.answer_evaluation.semantic_method,
                        evaluator_model=evaluation.answer_evaluation.evaluator_model,
                        evaluator_provider=evaluation.answer_evaluation.evaluator_provider,
                    )
                    if evaluation.answer_evaluation is not None
                    else None
                ),
                context_quality=(
                    AgentEvaluationContextQualityResponse(
                        budget_compliant=evaluation.context_quality.budget_compliant,
                        retrieval_evidence_present=(
                            evaluation.context_quality.retrieval_evidence_present
                        ),
                        has_semantic_memory_sources=(
                            evaluation.context_quality.has_semantic_memory_sources
                        ),
                        has_episodic_memory_sources=(
                            evaluation.context_quality.has_episodic_memory_sources
                        ),
                        has_working_memory_sources=(
                            evaluation.context_quality.has_working_memory_sources
                        ),
                        has_chat_history_sources=(
                            evaluation.context_quality.has_chat_history_sources
                        ),
                        has_tool_result_sources=(
                            evaluation.context_quality.has_tool_result_sources
                        ),
                        context_source_profile_changes=(
                            evaluation.context_quality.context_source_profile_changes
                        ),
                        assemblies_total=evaluation.context_quality.assemblies_total,
                        messages_total=evaluation.context_quality.messages_total,
                        estimated_tokens_total=(evaluation.context_quality.estimated_tokens_total),
                        estimated_tokens_max=evaluation.context_quality.estimated_tokens_max,
                        minimum_estimated_remaining_after_context=(
                            evaluation.context_quality.minimum_estimated_remaining_after_context
                        ),
                        source_counts=dict(evaluation.context_quality.source_counts),
                    )
                    if evaluation.context_quality is not None
                    else None
                ),
            )
            for evaluation in evaluations
        ],
        limit=limit,
    )


@router.get(
    "/runs/{run_id}/events",
    response_model=AgentRunEventListResponse,
)
async def list_agent_run_events(
    request: Request,
    run_id: str,
    limit: int = Query(default=100, ge=1, le=100),
    event_type: AgentExecutionEventType | None = None,
    step_id: str | None = None,
    attempt: int | None = Query(default=None, ge=1),
    provider: str | None = None,
    cursor: str | None = Query(default=None),
    service: AgentRunApplicationService = Depends(
        get_agent_run_application_service,
    ),
) -> AgentRunEventListResponse:
    try:
        page = service.list_events(
            run_id,
            tenant_id=getattr(request.state, "tenant_id", None),
            principal=getattr(request.state, "principal", None),
            event_type=event_type,
            step_id=step_id,
            attempt=attempt,
            provider=provider,
            limit=limit,
            cursor=cursor,
        )
    except AgentRunAccessDeniedError as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(exc),
        ) from exc
    except LookupError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    return AgentRunEventListResponse(
        events=[
            AgentRunEventResponse(
                event_type=event.event_type.value,
                agent_name=event.agent_name,
                run_id=event.run_id,
                session_id=event.session_id,
                user_id=event.user_id,
                tool_round=event.tool_round,
                tool_name=event.tool_name,
                call_id=event.call_id,
                step_id=event.step_id,
                step_index=event.step_index,
                step_name=event.step_name,
                attempt=event.attempt,
                provider=event.provider,
                model=event.model,
                metadata=event.metadata,
            )
            for event in page.events
        ],
        next_cursor=page.next_cursor,
        has_more=page.has_more,
    )
