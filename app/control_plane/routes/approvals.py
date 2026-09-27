from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from ai_platform.agents.models import AgentResponse
from app.control_plane.agent_runs.application_service import (
    AgentRunApplicationService,
)
from app.control_plane.agent_runs.exceptions import AgentRunAccessDeniedError
from app.control_plane.approvals.continuation_service import (
    AgentRunApprovalContinuationService,
)
from app.control_plane.approvals.models import ApprovalStatus
from app.control_plane.approvals.repository import ApprovalRequestRepository
from app.control_plane.dependencies import (
    get_agent_run_application_service,
    get_agent_run_approval_continuation_service,
    get_approval_request_repository,
)
from app.control_plane.schemas.approvals import (
    ApprovalDecisionRequest,
    ApprovalDecisionResponse,
    ApprovalInboxItem,
    ApprovalOverrideRequest,
)

router = APIRouter(
    prefix="/api/v1/approvals",
    tags=["approvals"],
)


def _authenticated_identity(request: Request) -> tuple[str, str]:
    tenant_id = getattr(request.state, "tenant_id", None)
    principal = getattr(request.state, "principal", None)

    if tenant_id is None or principal is None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Authenticated tenant and principal context are required.",
        )

    return tenant_id, principal


def _to_response(
    response: AgentResponse,
    *,
    run_id: str,
) -> ApprovalDecisionResponse:
    return ApprovalDecisionResponse(
        run_id=run_id,
        agent_name=response.agent_name,
        output=response.output,
        session_id=response.session_id,
        metadata=response.metadata,
    )


def _authorize_approval_access(
    *,
    request: Request,
    approval_id: str,
    approval_repository: ApprovalRequestRepository,
    agent_run_application_service: AgentRunApplicationService,
):
    tenant_id, principal = _authenticated_identity(request)

    approval = approval_repository.get(approval_id)

    if approval is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Approval request '{approval_id}' was not found.",
        )

    try:
        run = agent_run_application_service.get_run(
            approval.run_id,
            tenant_id=tenant_id,
            principal=principal,
        )
    except AgentRunAccessDeniedError as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(exc),
        ) from exc

    if run is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Approval request '{approval_id}' was not found.",
        )

    return principal, approval.run_id


@router.get(
    "",
    response_model=list[ApprovalInboxItem],
)
async def list_approvals(
    request: Request,
    approval_repository: ApprovalRequestRepository = Depends(
        get_approval_request_repository,
    ),
    approval_status: ApprovalStatus | None = Query(
        default=None,
        alias="status",
    ),
    limit: int = Query(default=100, ge=1, le=100),
) -> list[ApprovalInboxItem]:
    tenant_id, principal = _authenticated_identity(request)

    approvals = approval_repository.list(
        tenant_id=tenant_id,
        principal=principal,
        status=approval_status,
        limit=limit,
    )

    return [
        ApprovalInboxItem(
            approval_id=approval.approval_id,
            run_id=approval.run_id,
            step_id=approval.step_id,
            call_id=approval.call_id,
            tool_name=approval.tool_name,
            idempotency_key=approval.idempotency_key,
            status=approval.status.value,
            policy_name=approval.policy_name,
            policy_version=approval.policy_version,
            risk_tier=approval.risk_tier,
            requested_action=approval.requested_action,
            policy_metadata=approval.policy_metadata,
            resolved_by=approval.resolved_by,
            resolution_reason=approval.resolution_reason,
            created_at=approval.created_at,
            updated_at=approval.updated_at,
            resolved_at=approval.resolved_at,
        )
        for approval in approvals
    ]


@router.post(
    "/{approval_id}/approve",
    response_model=ApprovalDecisionResponse,
)
async def approve_approval(
    request: Request,
    approval_id: str,
    payload: ApprovalDecisionRequest | None = None,
    approval_repository: ApprovalRequestRepository = Depends(
        get_approval_request_repository,
    ),
    agent_run_application_service: AgentRunApplicationService = Depends(
        get_agent_run_application_service,
    ),
    service: AgentRunApprovalContinuationService = Depends(
        get_agent_run_approval_continuation_service,
    ),
) -> ApprovalDecisionResponse:
    principal, run_id = _authorize_approval_access(
        request=request,
        approval_id=approval_id,
        approval_repository=approval_repository,
        agent_run_application_service=agent_run_application_service,
    )

    try:
        response = await service.continue_approval(
            approval_id,
            status=ApprovalStatus.APPROVED,
            resolved_by=principal,
            resolution_reason=payload.reason if payload is not None else None,
        )
    except PermissionError as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
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

    return _to_response(response, run_id=run_id)


@router.post(
    "/{approval_id}/reject",
    response_model=ApprovalDecisionResponse,
)
async def reject_approval(
    request: Request,
    approval_id: str,
    payload: ApprovalDecisionRequest | None = None,
    approval_repository: ApprovalRequestRepository = Depends(
        get_approval_request_repository,
    ),
    agent_run_application_service: AgentRunApplicationService = Depends(
        get_agent_run_application_service,
    ),
    service: AgentRunApprovalContinuationService = Depends(
        get_agent_run_approval_continuation_service,
    ),
) -> ApprovalDecisionResponse:
    principal, run_id = _authorize_approval_access(
        request=request,
        approval_id=approval_id,
        approval_repository=approval_repository,
        agent_run_application_service=agent_run_application_service,
    )

    try:
        response = await service.continue_approval(
            approval_id,
            status=ApprovalStatus.REJECTED,
            resolved_by=principal,
            resolution_reason=payload.reason if payload is not None else None,
        )
    except PermissionError as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
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

    return _to_response(response, run_id=run_id)


@router.post(
    "/{approval_id}/override",
    response_model=ApprovalDecisionResponse,
)
async def override_approval(
    request: Request,
    approval_id: str,
    payload: ApprovalOverrideRequest,
    approval_repository: ApprovalRequestRepository = Depends(
        get_approval_request_repository,
    ),
    agent_run_application_service: AgentRunApplicationService = Depends(
        get_agent_run_application_service,
    ),
    service: AgentRunApprovalContinuationService = Depends(
        get_agent_run_approval_continuation_service,
    ),
) -> ApprovalDecisionResponse:
    principal, run_id = _authorize_approval_access(
        request=request,
        approval_id=approval_id,
        approval_repository=approval_repository,
        agent_run_application_service=agent_run_application_service,
    )

    try:
        response = await service.override_approval(
            approval_id,
            actor=principal,
            reason=payload.reason,
        )
    except PermissionError as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
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

    return _to_response(response, run_id=run_id)
