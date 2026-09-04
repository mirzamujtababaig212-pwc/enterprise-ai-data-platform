from __future__ import annotations

from fastapi import APIRouter

from app.control_plane.schemas.capabilities import (
    CapabilitiesResponse,
    Capability,
)
from app.control_plane.schemas.health import HealthResponse

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
