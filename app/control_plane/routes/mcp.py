from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from ai_platform.agents.policy import PolicyViolationError
from app.control_plane.dependencies import (
    get_mcp_management_service,
    get_mcp_server_lifecycle_service,
)
from app.control_plane.mcp_management_service import MCPManagementService
from app.control_plane.mcp_servers.api_mapping import (
    config_request_to_domain,
    domain_to_lifecycle_response,
    update_payload_to_domain_values,
)
from app.control_plane.mcp_servers.exceptions import (
    MCPServerAlreadyExistsError,
    MCPServerNotFoundError,
)
from app.control_plane.mcp_servers.lifecycle_service import (
    MCPServerLifecycleService,
)
from app.control_plane.mcp_servers.models import MCPServerDesiredState
from app.control_plane.schemas.mcp import (
    MCPSyncResponse,
    MCPServerConfigRequest,
    MCPServerCreateRequest,
    MCPServerHealthListResponse,
    MCPServerHealthResponse,
    MCPServerLifecycleListResponse,
    MCPServerLifecycleResponse,
    MCPServerListResponse,
    MCPServerResponse,
    MCPServerUpdateRequest,
    MCPToolResponse,
)

router = APIRouter(
    prefix="/api/v1/mcp",
    tags=["mcp"],
)


def _tenant_id(request: Request) -> str | None:
    return getattr(request.state, "tenant_id", None)


def _principal(request: Request) -> str | None:
    return getattr(request.state, "principal", None)


def _require_identity(request: Request) -> tuple[str, str]:
    tenant_id = _tenant_id(request)
    principal = _principal(request)

    if tenant_id is None or principal is None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Authenticated tenant and principal are required.",
        )

    return tenant_id, principal


def _parse_desired_state(value: str | None) -> MCPServerDesiredState | None:
    if value is None:
        return None

    try:
        return MCPServerDesiredState(value)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Invalid MCP desired state: {value}",
        ) from exc


def _config_from_create_payload(
    payload: MCPServerCreateRequest,
):
    config_payload = MCPServerConfigRequest(
        name=payload.name,
        transport=payload.transport,
        command=payload.command,
        args=payload.args,
        cwd=payload.cwd,
        url=payload.url,
        timeout=payload.timeout,
        read_timeout=payload.read_timeout,
        health_check_timeout=payload.health_check_timeout,
        verify_ssl=payload.verify_ssl,
        recovery_policy=payload.recovery_policy,
        tool_capabilities=payload.tool_capabilities,
    )

    return config_request_to_domain(config_payload)


@router.get(
    "/servers",
    response_model=MCPServerListResponse,
)
async def list_mcp_servers(
    request: Request,
    service: MCPManagementService = Depends(get_mcp_management_service),
) -> MCPServerListResponse:
    tenant_id, _ = _require_identity(request)

    try:
        servers = await service.list_servers(tenant_id)
    except PolicyViolationError as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(exc),
        ) from exc

    return MCPServerListResponse(
        servers=[
            MCPServerResponse(
                server_id=server.server_id,
                transport=server.transport,
                connected=server.connected,
            )
            for server in servers
        ],
    )


@router.get(
    "/servers/health",
    response_model=MCPServerHealthListResponse,
)
async def mcp_server_health(
    request: Request,
    service: MCPManagementService = Depends(get_mcp_management_service),
) -> MCPServerHealthListResponse:
    tenant_id, _ = _require_identity(request)

    try:
        health = await service.health(tenant_id)
    except PolicyViolationError as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(exc),
        ) from exc

    return MCPServerHealthListResponse(
        servers=[
            MCPServerHealthResponse(
                server_id=item.server_id,
                status=item.status,
                latency_ms=item.latency_ms,
                last_check=item.last_check,
                error=item.error,
            )
            for item in health
        ],
    )


@router.post(
    "/servers/{server_id}/sync",
    response_model=MCPSyncResponse,
)
async def sync_mcp_server(
    server_id: str,
    request: Request,
    service: MCPManagementService = Depends(get_mcp_management_service),
) -> MCPSyncResponse:
    tenant_id, _ = _require_identity(request)

    try:
        tools = await service.sync(
            tenant_id,
            server_id,
        )
    except PolicyViolationError as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(exc),
        ) from exc
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"MCP server '{server_id}' was not found.",
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc

    return MCPSyncResponse(
        server_id=server_id,
        tools=[
            MCPToolResponse(
                name=tool.name,
                description=tool.description,
                input_schema=tool.input_schema,
                metadata=tool.metadata,
                enabled=tool.enabled,
            )
            for tool in tools
        ],
    )


@router.post(
    "/managed-servers",
    response_model=MCPServerLifecycleResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_managed_mcp_server(
    payload: MCPServerCreateRequest,
    request: Request,
    service: MCPServerLifecycleService = Depends(get_mcp_server_lifecycle_service),
) -> MCPServerLifecycleResponse:
    tenant_id, _ = _require_identity(request)

    try:
        desired_state = MCPServerDesiredState(payload.desired_state)
        config = _config_from_create_payload(payload)

        server = await service.create(
            tenant_id=tenant_id,
            config=config,
            desired_state=desired_state,
            secret_references=payload.secret_references,
        )
    except MCPServerAlreadyExistsError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc

    return domain_to_lifecycle_response(server)


@router.get(
    "/managed-servers",
    response_model=MCPServerLifecycleListResponse,
)
async def list_managed_mcp_servers(
    request: Request,
    desired_state: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=1000),
    service: MCPServerLifecycleService = Depends(get_mcp_server_lifecycle_service),
) -> MCPServerLifecycleListResponse:
    tenant_id, _ = _require_identity(request)

    state = _parse_desired_state(desired_state)

    servers = service.list(
        tenant_id=tenant_id,
        desired_state=state,
        limit=limit,
    )

    return MCPServerLifecycleListResponse(
        servers=[domain_to_lifecycle_response(server) for server in servers],
    )


@router.get(
    "/managed-servers/{server_id}",
    response_model=MCPServerLifecycleResponse,
)
async def get_managed_mcp_server(
    server_id: str,
    request: Request,
    service: MCPServerLifecycleService = Depends(get_mcp_server_lifecycle_service),
) -> MCPServerLifecycleResponse:
    tenant_id, _ = _require_identity(request)

    try:
        server = service.get(
            tenant_id=tenant_id,
            server_id=server_id,
        )
    except MCPServerNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    return domain_to_lifecycle_response(server)


@router.patch(
    "/managed-servers/{server_id}",
    response_model=MCPServerLifecycleResponse,
)
async def update_managed_mcp_server(
    server_id: str,
    payload: MCPServerUpdateRequest,
    request: Request,
    service: MCPServerLifecycleService = Depends(get_mcp_server_lifecycle_service),
) -> MCPServerLifecycleResponse:
    tenant_id, _ = _require_identity(request)

    try:
        existing = service.get(
            tenant_id=tenant_id,
            server_id=server_id,
        )

        config, secret_references = update_payload_to_domain_values(
            payload,
            existing,
        )

        desired_state = (
            MCPServerDesiredState(payload.desired_state)
            if payload.desired_state is not None
            else None
        )

        server = await service.update(
            tenant_id=tenant_id,
            server_id=server_id,
            config=config,
            desired_state=desired_state,
            secret_references=secret_references,
        )
    except MCPServerNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except MCPServerAlreadyExistsError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc

    return domain_to_lifecycle_response(server)


@router.post(
    "/managed-servers/{server_id}/reconcile",
    response_model=MCPServerLifecycleResponse,
)
async def reconcile_managed_mcp_server(
    server_id: str,
    request: Request,
    service: MCPServerLifecycleService = Depends(get_mcp_server_lifecycle_service),
) -> MCPServerLifecycleResponse:
    tenant_id, _ = _require_identity(request)

    try:
        server = await service.reconcile(
            tenant_id=tenant_id,
            server_id=server_id,
        )
    except MCPServerNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc

    return domain_to_lifecycle_response(server)


@router.delete(
    "/managed-servers/{server_id}",
    response_model=MCPServerLifecycleResponse,
)
async def delete_managed_mcp_server(
    server_id: str,
    request: Request,
    service: MCPServerLifecycleService = Depends(get_mcp_server_lifecycle_service),
) -> MCPServerLifecycleResponse:
    tenant_id, _ = _require_identity(request)

    try:
        server = await service.delete(
            tenant_id=tenant_id,
            server_id=server_id,
        )
    except MCPServerNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc

    return domain_to_lifecycle_response(server)
