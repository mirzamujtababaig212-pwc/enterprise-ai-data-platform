from __future__ import annotations

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request, status

from ai_platform.agents.models import AgentRequest

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
    get_agent_run_application_service,
    get_agent_run_recovery_service,
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


@router.get(
    "/runs/{run_id}/events",
    response_model=AgentRunEventListResponse,
)
async def list_agent_run_events(
    request: Request,
    run_id: str,
    limit: int = Query(default=100, ge=1, le=100),
    service: AgentRunApplicationService = Depends(
        get_agent_run_application_service,
    ),
) -> AgentRunEventListResponse:
    try:
        events = service.list_events(
            run_id,
            tenant_id=getattr(request.state, "tenant_id", None),
            principal=getattr(request.state, "principal", None),
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
                provider=event.provider,
                model=event.model,
                metadata=event.metadata,
            )
            for event in events
        ]
    )
