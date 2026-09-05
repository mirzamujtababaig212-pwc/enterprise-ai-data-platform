from __future__ import annotations

import time
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, Request
from uuid import uuid4
from ai_platform.llm_gateway.auth.security import api_key_header
from ai_platform.llm_gateway.exceptions.gateway_exceptions import ProviderNotFound
from ai_platform.llm_gateway.models.chat import ChatRequest, ChatResponse
from ai_platform.llm_gateway.models.embedding import (
    EmbeddingRequest,
    EmbeddingResponse,
)
from ai_platform.llm_gateway.models.usage import UsageMetrics

from app.control_plane.dependencies import (
    get_llm_router,
    get_usage_store,
)
from app.control_plane.usage.models import UsageEvent
from ai_platform.llm_gateway.gateway.cost import build_usage_record
from ai_platform.llm_gateway.exceptions.provider_exceptions import (
    ProviderRateLimitError,
)

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
    usage_store=Depends(get_usage_store),
) -> ChatResponse:
    start = time.time()
    llm_router = get_llm_router()

    request_id = getattr(
        http_request.state,
        "request_id",
        None,
    )

    if request_id is None:
        request_id = str(uuid4())

    try:
        result = await llm_router.route_chat_with_metadata(
            request.model_dump(),
        )
        reply = result.response
    except ProviderNotFound as exc:
        usage_store.record(
            UsageEvent(
                request_id=request_id,
                capability="llm.chat",
                provider=request.provider,
                model=request.model,
                latency_ms=(time.time() - start) * 1000,
                status="error",
            )
        )
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc
    except ProviderRateLimitError as exc:
        usage_store.record(
            UsageEvent(
                request_id=request_id,
                capability="llm.chat",
                provider=request.provider,
                model=request.model,
                latency_ms=(time.time() - start) * 1000,
                status="error",
            )
        )
        raise HTTPException(
            status_code=429,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        usage_store.record(
            UsageEvent(
                request_id=request_id,
                capability="llm.chat",
                provider=request.provider,
                model=request.model,
                latency_ms=(time.time() - start) * 1000,
                status="error",
            )
        )
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

    provider = result.provider_name
    model = request.model

    usage_record = build_usage_record(
        provider=provider,
        model=model,
        prompt_tokens=tokens_in,
        completion_tokens=tokens_out,
        request_id=request_id,
    )

    latency_ms = (time.time() - start) * 1000

    usage_event = UsageEvent(
        request_id=request_id,
        capability="llm.chat",
        provider=usage_record.provider,
        model=usage_record.model,
        tokens_in=usage_record.prompt_tokens,
        tokens_out=usage_record.completion_tokens,
        estimated_cost=float(usage_record.estimated_cost_usd),
        latency_ms=latency_ms,
        status="success",
    )

    usage_store.record(usage_event)

    metrics = UsageMetrics(
        request_id=request_id,
        timestamp=datetime.now(UTC),
        latency_ms=int(latency_ms),
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        estimated_cost=float(usage_record.estimated_cost_usd),
        status="success",
    )

    http_request.state.metrics = metrics.model_dump()

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
    usage_store=Depends(get_usage_store),
) -> EmbeddingResponse:
    start = time.time()
    llm_router = get_llm_router()

    request_id = getattr(
        http_request.state,
        "request_id",
        None,
    )

    if request_id is None:
        request_id = UsageMetrics().request_id

    try:
        result = await llm_router.route_embeddings_with_metadata(
            request.model_dump(),
        )
    except ProviderNotFound as exc:
        usage_store.record(
            UsageEvent(
                request_id=request_id,
                capability="llm.embeddings",
                provider=request.provider,
                model=request.model,
                latency_ms=(time.time() - start) * 1000,
                status="error",
            )
        )

        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        usage_store.record(
            UsageEvent(
                request_id=request_id,
                capability="llm.embeddings",
                provider=request.provider,
                model=request.model,
                latency_ms=(time.time() - start) * 1000,
                status="error",
            )
        )

        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc

    vector = result.response
    provider = result.provider_name
    model = request.model

    tokens_in = len(request.text.split())
    tokens_out = 0

    usage_record = build_usage_record(
        provider=provider,
        model=model,
        prompt_tokens=tokens_in,
        completion_tokens=tokens_out,
        request_id=request_id,
    )

    latency_ms = (time.time() - start) * 1000

    usage_store.record(
        UsageEvent(
            request_id=request_id,
            capability="llm.embeddings",
            provider=usage_record.provider,
            model=usage_record.model,
            tokens_in=usage_record.prompt_tokens,
            tokens_out=usage_record.completion_tokens,
            estimated_cost=float(
                usage_record.estimated_cost_usd,
            ),
            latency_ms=latency_ms,
            status="success",
        )
    )

    metrics = UsageMetrics(
        request_id=request_id,
        timestamp=datetime.now(UTC),
        latency_ms=int(latency_ms),
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        estimated_cost=float(
            usage_record.estimated_cost_usd,
        ),
        status="success",
    )

    http_request.state.metrics = metrics.model_dump()

    return EmbeddingResponse(
        vector=vector,
        metrics=metrics,
    )
