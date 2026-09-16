from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status

from ai_platform.agents.models import AgentRequest

from app.control_plane.agent_runs.application_service import (
    AgentRunApplicationService,
)
from app.control_plane.agent_runs.models import AgentRunStatus
from app.control_plane.dependencies import get_agent_run_application_service
from app.control_plane.schemas.agents import (
    AgentRunDetailResponse,
    AgentRunListResponse,
    AgentRunRequest,
    AgentRunResponse,
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
    agent_name: str,
    payload: AgentRunRequest,
    service: AgentRunApplicationService = Depends(
        get_agent_run_application_service,
    ),
) -> AgentRunResponse:
    try:
        request = AgentRequest(
            input=payload.input,
            session_id=payload.session_id,
            user_id=payload.user_id,
            metadata=payload.metadata,
        )

        response = await service.execute(
            agent_name=agent_name,
            request=request,
        )

    except LookupError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
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
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid agent run status: {run_status}",
            ) from exc

    runs = service.list_runs(
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


@router.get(
    "/runs/{run_id}",
    response_model=AgentRunDetailResponse,
)
async def get_agent_run(
    run_id: str,
    service: AgentRunApplicationService = Depends(
        get_agent_run_application_service,
    ),
) -> AgentRunDetailResponse:
    run = service.get_run(run_id)

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
