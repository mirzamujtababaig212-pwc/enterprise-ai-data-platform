from fastapi import Request
from fastapi.responses import JSONResponse
from opentelemetry import trace

from ai_platform.llm_gateway.config.settings import settings

tracer = trace.get_tracer(__name__)

API_KEY_NAME = "x-api-key"

PUBLIC_PATHS = {
    "/api/v1/health",
    "/api/v1/platform/health",
    "/docs",
    "/redoc",
    "/openapi.json",
}


def get_valid_api_keys() -> set[str]:
    return {key.strip() for key in settings.API_KEY.split(",") if key.strip()}


class ControlPlaneAPIKeyMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request = Request(scope, receive)
        path = request.url.path

        if path in PUBLIC_PATHS:
            await self.app(scope, receive, send)
            return

        api_key = request.headers.get(API_KEY_NAME)
        valid_api_keys = get_valid_api_keys()

        if api_key not in valid_api_keys:
            response = JSONResponse(
                status_code=401,
                content={
                    "detail": "Invalid or missing API key",
                },
                headers={
                    "WWW-Authenticate": "ApiKey",
                },
            )
            await response(scope, receive, send)
            return

        with tracer.start_as_current_span("control_plane.authentication"):
            await self.app(scope, receive, send)
