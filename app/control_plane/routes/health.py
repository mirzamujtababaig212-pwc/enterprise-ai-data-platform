from __future__ import annotations

from fastapi import APIRouter

from app.control_plane.schemas.health import HealthResponse

router = APIRouter(
    prefix="/api/v1",
    tags=["health"],
)


@router.get(
    "/health",
    response_model=HealthResponse,
)
async def health() -> HealthResponse:
    return HealthResponse(
        status="healthy",
        service="enterprise-ai-control-plane",
        version="1.0.0",
    )
