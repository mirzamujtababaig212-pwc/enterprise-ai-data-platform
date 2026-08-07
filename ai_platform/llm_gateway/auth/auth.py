from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from opentelemetry import trace

tracer = trace.get_tracer(__name__)

API_KEY_NAME = "x-api-key"
VALID_API_KEYS = {"super-secret-key"}


class APIKeyMiddleware:

    def __init__(self, app: FastAPI):
        self.app = app

    async def __call__(self, scope, receive, send):

        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request = Request(scope, receive)

        path = request.url.path

        if path in (
            "/docs",
            "/redoc",
            "/openapi.json",
            "/favicon.ico",
            "/metrics",
        ):
            await self.app(scope, receive, send)
            return

        api_key = request.headers.get(API_KEY_NAME)

        if api_key not in VALID_API_KEYS:
            response = JSONResponse(
                status_code=401,
                content={
                    "detail": "Invalid or missing API key",
                },
            )
            await response(scope, receive, send)
            return

        with tracer.start_as_current_span("authentication"):
            await self.app(scope, receive, send)
