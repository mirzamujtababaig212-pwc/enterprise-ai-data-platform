from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

API_KEY_NAME = "x-api-key"
VALID_API_KEYS = {"super-secret-key"}  # Replace with your real keys


class APIKeyMiddleware:
    def __init__(self, app: FastAPI):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            request = Request(scope, receive)
            api_key = request.headers.get(API_KEY_NAME)

            if api_key not in VALID_API_KEYS:
                response = JSONResponse(
                    status_code=401,
                    content={"detail": "Invalid or missing API key"},
                )
                await response(scope, receive, send)
                return

        await self.app(scope, receive, send)
