from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from ai_platform.agents.models import AgentRequest
from ai_platform.agents.runtime import AgentRuntime

from app.control_plane.dependencies import get_agent_runtime
from app.control_plane.schemas.agents import (
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
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> AgentRunResponse:
    try:
        request = AgentRequest(
            input=payload.input,
            session_id=payload.session_id,
            user_id=payload.user_id,
            metadata=payload.metadata,
        )

        response = await runtime.run(
            agent_name,
            request,
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
        agent_name=response.agent_name,
        output=response.output,
        session_id=response.session_id,
        metadata=response.metadata,
    )
