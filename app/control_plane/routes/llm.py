from __future__ import annotations

import time
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, Request

from ai_platform.llm_gateway.auth.security import api_key_header
from ai_platform.llm_gateway.exceptions.gateway_exceptions import ProviderNotFound
from ai_platform.llm_gateway.models.chat import ChatRequest, ChatResponse
from ai_platform.llm_gateway.models.embedding import (
    EmbeddingRequest,
    EmbeddingResponse,
)
from ai_platform.llm_gateway.models.usage import UsageMetrics

from app.control_plane.dependencies import get_llm_router

router = APIRouter(
    prefix="/api/v1/llm",
    tags=["llm"],
)


@router.post(
    "/chat",
    response_model=ChatResponse,
    dependencies=[Depends(api_key_header)],
)
async def chat(
    http_request: Request,
    request: ChatRequest,
) -> ChatResponse:
    start = time.time()
    llm_router = get_llm_router()

    try:
        reply = await llm_router.route_chat(request.model_dump())
    except ProviderNotFound as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    usage = reply.get("usage", {})

    tokens_in = usage.get(
        "tokens_in",
        len(request.prompt.split()),
    )
    tokens_out = usage.get(
        "tokens_out",
        len(reply.get("reply", "").split()),
    )

    request_id = getattr(
        http_request.state,
        "request_id",
        None,
    )

    metrics = UsageMetrics(
        request_id=request_id or UsageMetrics().request_id,
        timestamp=datetime.now(UTC),
        latency_ms=int((time.time() - start) * 1000),
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        estimated_cost=0.0,
        status="success",
    )

    return ChatResponse(
        reply=reply["reply"],
        metrics=metrics,
    )


@router.post(
    "/embeddings",
    response_model=EmbeddingResponse,
    dependencies=[Depends(api_key_header)],
)
async def embeddings(
    http_request: Request,
    request: EmbeddingRequest,
) -> EmbeddingResponse:
    start = time.time()
    llm_router = get_llm_router()

    try:
        vector = await llm_router.route_embeddings(request.model_dump())
    except ProviderNotFound as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc

    request_id = getattr(
        http_request.state,
        "request_id",
        None,
    )

    metrics = UsageMetrics(
        request_id=request_id or UsageMetrics().request_id,
        timestamp=datetime.now(UTC),
        latency_ms=int((time.time() - start) * 1000),
        tokens_in=len(request.text.split()),
        tokens_out=len(vector),
        estimated_cost=0.002,
        status="success",
    )

    return EmbeddingResponse(
        vector=vector,
        metrics=metrics,
    )
