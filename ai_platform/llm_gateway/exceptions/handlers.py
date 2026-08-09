from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from ai_platform.llm_gateway.exceptions.error_models import (
    ErrorDetail,
    ErrorResponse,
)
from ai_platform.llm_gateway.exceptions.provider_exceptions import (
    ProviderAuthenticationError,
    ProviderConnectionError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)

from .gateway_exceptions import ProviderNotFound


async def provider_not_found_handler(
    request: Request,
    exc: ProviderNotFound,
):
    return JSONResponse(
        status_code=400,
        content={"error": str(exc)},
    )


async def http_exception_handler(
    request: Request,
    exc: StarletteHTTPException,
):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=ErrorDetail(
                code=exc.status_code,
                message=exc.detail,
            ),
            request_id=getattr(request.state, "request_id", None),
        ).model_dump(),
    )


async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
):
    return JSONResponse(
        status_code=422,
        content=ErrorResponse(
            error=ErrorDetail(
                code=422,
                message="Request validation failed",
            ),
            request_id=getattr(request.state, "request_id", None),
        ).model_dump(),
    )


async def provider_auth_handler(
    request: Request,
    exc: ProviderAuthenticationError,
):
    return JSONResponse(
        status_code=401,
        content={
            "status": "error",
            "error": {"code": 401, "message": str(exc)},
            "request_id": getattr(request.state, "request_id", None),
        },
    )


async def provider_rate_limit_handler(
    request: Request,
    exc: ProviderRateLimitError,
):
    return JSONResponse(
        status_code=429,
        content={
            "status": "error",
            "error": {
                "code": 429,
                "message": str(exc),
            },
            "request_id": request.state.request_id,
        },
    )


async def provider_timeout_handler(
    request: Request,
    exc: ProviderTimeoutError,
):
    return JSONResponse(
        status_code=504,
        content={
            "status": "error",
            "error": {"code": 504, "message": str(exc)},
            "request_id": getattr(request.state, "request_id", None),
        },
    )


async def provider_connection_handler(
    request: Request,
    exc: ProviderConnectionError,
):
    return JSONResponse(
        status_code=503,
        content={
            "status": "error",
            "error": {"code": 503, "message": str(exc)},
            "request_id": getattr(request.state, "request_id", None),
        },
    )
