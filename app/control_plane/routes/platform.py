from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from app.control_plane.dependencies import get_usage_store
from app.control_plane.schemas.capabilities import (
    CapabilitiesResponse,
    Capability,
)
from app.control_plane.schemas.health import HealthResponse
from app.control_plane.schemas.usage import (
    UsageEventResponse,
    UsageResponse,
)

router = APIRouter(
    prefix="/api/v1/platform",
    tags=["platform"],
)


@router.get(
    "/health",
    response_model=HealthResponse,
)
async def platform_health() -> HealthResponse:
    return HealthResponse(
        status="healthy",
        service="enterprise-ai-platform",
        version="1.0.0",
    )


@router.get(
    "/capabilities",
    response_model=CapabilitiesResponse,
)
async def capabilities() -> CapabilitiesResponse:
    return CapabilitiesResponse(
        service="enterprise-ai-platform",
        capabilities=[
            Capability(
                name="llm.chat",
                description="Enterprise LLM chat routing with provider fallback.",
                status="available",
            ),
            Capability(
                name="llm.embeddings",
                description="Enterprise embedding generation through the LLM Gateway.",
                status="available",
            ),
            Capability(
                name="ml.vehicle-risk",
                description="Vehicle risk inference through the MLflow champion model.",
                status="available",
            ),
            Capability(
                name="rag.index",
                description="Enterprise document indexing with chunking and vector embeddings.",
                status="available",
            ),
            Capability(
                name="rag.query",
                description="Enterprise retrieval-augmented generation with semantic retrieval and LLM generation.",
                status="available",
            ),
        ],
    )


@router.get(
    "/usage",
    response_model=UsageResponse,
)
async def usage(
    request_id: str | None = Query(default=None),
    capability: str | None = Query(default=None),
    status: str | None = Query(default=None),
    limit: int = Query(default=100, gt=0, le=1000),
    usage_store=Depends(get_usage_store),
) -> UsageResponse:
    events = usage_store.list(
        request_id=request_id,
        capability=capability,
        status=status,
        limit=limit,
    )

    return UsageResponse(
        events=[UsageEventResponse.model_validate(event) for event in events],
        count=len(events),
    )
