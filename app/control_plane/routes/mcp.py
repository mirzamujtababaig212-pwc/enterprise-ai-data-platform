from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.control_plane.dependencies import get_mcp_management_service
from app.control_plane.mcp_management_service import MCPManagementService
from app.control_plane.schemas.mcp import (
    MCPSyncResponse,
    MCPServerHealthListResponse,
    MCPServerHealthResponse,
    MCPServerListResponse,
    MCPServerResponse,
    MCPToolResponse,
)
from ai_platform.agents.policy import PolicyViolationError

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
