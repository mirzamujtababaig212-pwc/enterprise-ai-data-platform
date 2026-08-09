from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from ai_platform.llm_gateway.logging.logger import get_logger

logger = get_logger()


class ExceptionLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self,
        request: Request,
        call_next,
    ):
        try:
            response = await call_next(request)
            return response

        except Exception as exc:
            request_id = getattr(
                request.state,
                "request_id",
                None,
            )

            logger.exception(
                "Unhandled exception",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "endpoint": request.url.path,
                    "client_ip": (request.client.host if request.client else "unknown"),
                    "exception_type": type(exc).__name__,
                },
            )

            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal Server Error",
                    "request_id": request_id,
                },
            )
