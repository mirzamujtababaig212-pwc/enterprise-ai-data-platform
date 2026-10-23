from __future__ import annotations

import time
import logging
from datetime import UTC, datetime
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from ai_platform.agents.policy import PolicyViolationError
from ai_platform.llm_gateway.auth.security import api_key_header
from ai_platform.llm_gateway.exceptions.gateway_exceptions import ProviderNotFound
from ai_platform.llm_gateway.exceptions.provider_exceptions import (
    ProviderRateLimitError,
)
from ai_platform.llm_gateway.gateway.cost import build_usage_record
from ai_platform.llm_gateway.models.chat import ChatRequest, ChatResponse
from ai_platform.llm_gateway.models.embedding import (
    EmbeddingRequest,
    EmbeddingResponse,
)
from ai_platform.llm_gateway.models.usage import UsageMetrics
from ai_platform.llm_gateway.observability.prometheus import (
    ESTIMATED_COST_TOTAL,
    INPUT_TOKENS_TOTAL,
    MODEL_REQUESTS_TOTAL,
    OUTPUT_TOKENS_TOTAL,
)
from app.control_plane.dependencies import (
    get_llm_router,
    get_tenant_policy_engine,
    get_usage_store,
    tenant_policy_enforcement_required,
)
from app.control_plane.usage.models import UsageEvent

router = APIRouter(
    prefix="/api/v1/llm",
    tags=["llm"],
)

logger = logging.getLogger(__name__)


def _authorized_provider_allow_list(
    http_request: Request,
    model: str,
    provider: str | None,
) -> tuple[str, ...] | None:
    """Authorize a direct model request using authenticated tenant identity."""
    engine = get_tenant_policy_engine()
    if engine is None:
        if tenant_policy_enforcement_required():
            raise HTTPException(
                status_code=503,
                detail="Tenant policy engine is unavailable.",
            )
        return None

    tenant_id = getattr(http_request.state, "tenant_id", None)
    if not isinstance(tenant_id, str) or not tenant_id.strip():
        raise HTTPException(
            status_code=403,
            detail="Authenticated tenant identity is required.",
        )

    try:
        engine.authorize_model(
            tenant_id=tenant_id,
            model=model,
            provider=provider,
        )
        policy = engine.get_policy(tenant_id)
    except PolicyViolationError as exc:
        raise HTTPException(
            status_code=403,
            detail=str(exc),
        ) from exc

    if policy.allowed_providers is None:
        return None

    return tuple(sorted(policy.allowed_providers))


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
    allowed_providers = _authorized_provider_allow_list(
        http_request, request.model, request.provider
    )
    llm_router = get_llm_router()

    request_id = getattr(
        http_request.state,
        "request_id",
        None,
    )

    if request_id is None:
        request_id = str(uuid4())

    MODEL_REQUESTS_TOTAL.labels(
        model=request.model,
    ).inc()

    try:
        route_kwargs = (
            {"allowed_providers": allowed_providers} if allowed_providers is not None else {}
        )
        result = await llm_router.route_chat_with_metadata(
            request.model_dump(),
            **route_kwargs,
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
    pricing_model = result.model_name or model

    usage_record = build_usage_record(
        provider=provider,
        model=pricing_model,
        prompt_tokens=tokens_in,
        completion_tokens=tokens_out,
        request_id=request_id,
    )

    INPUT_TOKENS_TOTAL.inc(
        usage_record.prompt_tokens,
    )
    OUTPUT_TOKENS_TOTAL.inc(
        usage_record.completion_tokens,
    )
    ESTIMATED_COST_TOTAL.inc(
        float(usage_record.estimated_cost_usd),
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
    "/chat/stream",
    dependencies=[Depends(api_key_header)],
)
async def chat_stream(
    http_request: Request,
    request: ChatRequest,
) -> StreamingResponse:
    """Stream model output after tenant-policy authorization."""
    if not request.stream:
        raise HTTPException(
            status_code=400,
            detail="Set stream=true to use the streaming endpoint.",
        )

    # Authorize before returning StreamingResponse so policy failures
    # produce proper HTTP status codes instead of post-header SSE errors.
    allowed_providers = _authorized_provider_allow_list(
        http_request,
        request.model,
        request.provider,
    )

    llm_router = get_llm_router()
    route_kwargs = {"allowed_providers": allowed_providers} if allowed_providers is not None else {}

    async def generate():
        try:
            async for chunk in llm_router.route_stream(
                request.model_dump(),
                **route_kwargs,
            ):
                yield f"data: {chunk}\n\n"

            yield "data: [DONE]\n\n"

        except Exception:
            logger.exception("Control Plane LLM streaming failed.")
            yield ("event: error\n" 'data: {"error":"Stream generation failed."}\n\n')

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
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
    allowed_providers = _authorized_provider_allow_list(
        http_request, request.model, request.provider
    )
    llm_router = get_llm_router()

    request_id = getattr(
        http_request.state,
        "request_id",
        None,
    )

    if request_id is None:
        request_id = UsageMetrics().request_id

    MODEL_REQUESTS_TOTAL.labels(
        model=request.model,
    ).inc()

    try:
        route_kwargs = (
            {"allowed_providers": allowed_providers} if allowed_providers is not None else {}
        )
        result = await llm_router.route_embeddings_with_metadata(
            request.model_dump(),
            **route_kwargs,
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

    INPUT_TOKENS_TOTAL.inc(
        usage_record.prompt_tokens,
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
