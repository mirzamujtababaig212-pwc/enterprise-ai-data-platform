from fastapi import Body, FastAPI, HTTPException, Request
from datetime import datetime, timezone
import time
from ai_platform.llm_gateway.models.chat import (
    ChatRequest,
    ChatResponse,
)

from ai_platform.llm_gateway.models.embedding import (
    EmbeddingRequest,
    EmbeddingResponse,
)

from ai_platform.llm_gateway.models.health import (
    HealthResponse,
)
from ai_platform.llm_gateway.auth.auth import APIKeyMiddleware
from ai_platform.llm_gateway.metrics.metrics import Metrics
from ai_platform.llm_gateway.routing.router import router
from ai_platform.llm_gateway.middleware.request_id import (
    RequestIDMiddleware,
)
from ai_platform.llm_gateway.middleware.logging import (
    LoggingMiddleware,
)
from ai_platform.llm_gateway.middleware.timing import (
    TimingMiddleware,
)
from ai_platform.llm_gateway.config.settings import settings
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from ai_platform.llm_gateway.exceptions.gateway_exceptions import (
    ProviderNotFound,
)

from ai_platform.llm_gateway.exceptions.provider_exceptions import (
    ProviderAuthenticationError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderConnectionError,
)

from ai_platform.llm_gateway.exceptions.handlers import (
    provider_not_found_handler,
    http_exception_handler,
    validation_exception_handler,
    provider_auth_handler,
    provider_rate_limit_handler,
    provider_timeout_handler,
    provider_connection_handler,
)

from ai_platform.llm_gateway.logging.logger import get_logger

from ai_platform.llm_gateway.middleware.request_logging import (
    RequestLoggingMiddleware,
)

from ai_platform.llm_gateway.middleware.exception_logging import (
    ExceptionLoggingMiddleware,
)

from fastapi.responses import (
    Response,
    StreamingResponse,
)

from prometheus_client import (
    generate_latest,
    CONTENT_TYPE_LATEST,
)

from ai_platform.llm_gateway.tracing.tracing import (
    configure_tracing,
)

logger = get_logger()

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="""
Enterprise AI Platform

A production-ready LLM Gateway providing:

- Multi-provider routing
- Authentication
- Metrics
- Observability
- Embeddings
- Chat Completion
- Health Monitoring
""",
    contact={
        "name": "Enterprise AI Team",
        "email": "support@enterprise-ai.local",
    },
    license_info={
        "name": "Apache 2.0",
    },
)
logger.info(
    "LLM Gateway starting",
    extra={
        "component": "gateway",
        "version": settings.APP_VERSION,
    },
)
configure_tracing(app)
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(RequestIDMiddleware)
app.add_middleware(TimingMiddleware)
app.add_middleware(LoggingMiddleware)
app.add_middleware(APIKeyMiddleware)
app.add_middleware(ExceptionLoggingMiddleware)

app.add_exception_handler(
    ProviderNotFound,
    provider_not_found_handler,
)

app.add_exception_handler(
    StarletteHTTPException,
    http_exception_handler,
)

app.add_exception_handler(
    RequestValidationError,
    validation_exception_handler,
)

app.add_exception_handler(
    ProviderAuthenticationError,
    provider_auth_handler,
)

app.add_exception_handler(
    ProviderRateLimitError,
    provider_rate_limit_handler,
)

app.add_exception_handler(
    ProviderTimeoutError,
    provider_timeout_handler,
)

app.add_exception_handler(
    ProviderConnectionError,
    provider_connection_handler,
)

# Configure logging
chat_tag = ["Chat"]
embedding_tag = ["Embeddings"]
health_tag = ["Health"]
models_tag = ["Models"]


# Endpoints
@app.post(
    "/v1/chat",
    response_model=ChatResponse,
    tags=chat_tag,
    summary="Generate chat completion",
    description="Routes a prompt to the selected LLM provider and returns the generated response.",
)
async def chat_endpoint(
    http_request: Request,
    request: ChatRequest,
):
    start = time.time()
    logger.info(
        "Received chat request",
        extra={
            "provider": request.provider,
            "model": request.model,
            "endpoint": str(http_request.url.path),
            "method": http_request.method,
            "stream": request.stream,
        },
    )

    request_data = request.model_dump()

    if request.stream:

        async def generate():
            try:
                async for chunk in router.route_stream(request_data):
                    yield f"data: {chunk}\n\n"

                yield "data: [DONE]\n\n"

            except Exception:
                logger.exception("Streaming request failed.")
                raise

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "X-Accel-Buffering": "no",
                "Cache-Control": "no-cache",
            },
        )

    try:
        reply = await router.route_chat(request_data)

        usage = reply.get("usage", {})

        tokens_in = usage.get(
            "tokens_in",
            len(request.prompt.split()),
        )

        tokens_out = usage.get(
            "tokens_out",
            len(reply.get("reply", "").split()),
        )

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e),
        )

    metrics = Metrics(
        request_id=http_request.state.request_id,
        timestamp=datetime.now(timezone.utc),
        latency_ms=int((time.time() - start) * 1000),
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        estimated_cost=0.0,  # placeholder
        status="success",
    )
    http_request.state.metrics = metrics.model_dump()
    return ChatResponse(
        reply=reply["reply"],
        metrics=metrics,
    )


@app.post(
    "/v1/embeddings",
    response_model=EmbeddingResponse,
    tags=embedding_tag,
    summary="Generate embeddings",
    description="Creates vector embeddings for the supplied text.",
)
async def embeddings_endpoint(
    http_request: Request,
    request: EmbeddingRequest,
):
    start = time.time()
    try:
        vector = await router.route_embeddings(request.model_dump())
    except ValueError as e:
        raise HTTPException(
            status_code=404,
            detail=str(e),
        )

    metrics = Metrics(
        request_id=http_request.state.request_id,
        timestamp=datetime.now(timezone.utc),
        latency_ms=int((time.time() - start) * 1000),
        tokens_in=len(request.text.split()),
        tokens_out=len(vector),
        estimated_cost=0.002,  # placeholder
        status="success",
    )
    http_request.state.metrics = metrics.model_dump()
    print(http_request.state.metrics)

    return EmbeddingResponse(vector=vector, metrics=metrics)


@app.get(
    "/v1/models",
    tags=models_tag,
    summary="List available models",
    description="Returns all configured providers and their supported models.",
)
async def list_models():
    return await router.route_models()


@app.get(
    "/v1/health",
    response_model=HealthResponse,
    tags=health_tag,
    summary="Health Check",
    description="Returns gateway and provider health status.",
)
async def health_check():
    return HealthResponse(
        status="ok",
        providers=await router.route_health(),
    )


@app.get("/v1/test-error")
async def test_error():
    raise RuntimeError("This is a test exception")


@app.post("/v1/test-body")
async def test_body(payload: dict = Body(...)):
    """
    Temporary endpoint used to verify middleware body sanitization.
    """
    return {
        "received": payload,
    }


@app.get(
    "/metrics",
    include_in_schema=False,
)
async def metrics():

    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )
