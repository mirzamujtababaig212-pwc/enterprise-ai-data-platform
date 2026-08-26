from __future__ import annotations

import time
import uuid

from fastapi import APIRouter, HTTPException

from app.gateway.router import ProviderRouter
from app.providers.base import LLMRequest
from app.schemas.chat import (
    ChatCompletionRequest,
    ChatCompletionResponse,
)

router = APIRouter()


def build_router() -> ProviderRouter:
    from app.main import provider_router

    return provider_router


@router.post(
    "/v1/chat/completions",
    response_model=ChatCompletionResponse,
)
async def chat_completion(
    request: ChatCompletionRequest,
) -> ChatCompletionResponse:

    request_id = str(uuid.uuid4())

    started = time.perf_counter()

    llm_request = LLMRequest(
        messages=[
            {
                "role": message.role,
                "content": message.content,
            }
            for message in request.messages
        ],
        model=request.model,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
    )

    try:
        response = await build_router().generate(
            llm_request,
            provider_name=request.provider,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail="LLM provider request failed",
        ) from exc

    elapsed_ms = (time.perf_counter() - started) * 1000

    return ChatCompletionResponse(
        request_id=request_id,
        provider=response.provider,
        model=response.model,
        content=response.content,
        usage={
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "total_tokens": response.total_tokens,
        },
        metadata={
            "provider_request_id": response.request_id,
            "latency_ms": round(elapsed_ms, 2),
        },
    )
