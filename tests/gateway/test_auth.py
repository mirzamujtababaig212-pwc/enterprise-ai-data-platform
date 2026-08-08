import json

import pytest

from ai_platform.llm_gateway.auth.auth import APIKeyMiddleware


class DummyApp:
    def __init__(self):
        self.called = False

    async def __call__(self, scope, receive, send):
        self.called = True
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [],
            }
        )
        await send(
            {
                "type": "http.response.body",
                "body": b"OK",
            }
        )


@pytest.mark.asyncio
async def test_valid_api_key():

    app = DummyApp()
    middleware = APIKeyMiddleware(app)

    messages = []

    async def receive():
        return {"type": "http.request"}

    async def send(message):
        messages.append(message)

    scope = {
        "type": "http",
        "method": "GET",
        "path": "/v1/chat",
        "headers": [
            (b"x-api-key", b"super-secret-key"),
        ],
    }

    await middleware(scope, receive, send)

    assert app.called is True
    assert messages[0]["status"] == 200


@pytest.mark.asyncio
async def test_missing_api_key():

    app = DummyApp()
    middleware = APIKeyMiddleware(app)

    messages = []

    async def receive():
        return {"type": "http.request"}

    async def send(message):
        messages.append(message)

    scope = {
        "type": "http",
        "method": "GET",
        "path": "/v1/chat",
        "headers": [],
    }

    await middleware(scope, receive, send)

    assert app.called is False
    assert messages[0]["status"] == 401

    body = json.loads(messages[1]["body"])

    assert body["detail"] == "Invalid or missing API key"


@pytest.mark.asyncio
async def test_metrics_bypass():

    app = DummyApp()
    middleware = APIKeyMiddleware(app)

    messages = []

    async def receive():
        return {"type": "http.request"}

    async def send(message):
        messages.append(message)

    scope = {
        "type": "http",
        "method": "GET",
        "path": "/metrics",
        "headers": [],
    }

    await middleware(scope, receive, send)

    assert app.called is True
    assert messages[0]["status"] == 200


@pytest.mark.asyncio
async def test_non_http_scope():

    app = DummyApp()
    middleware = APIKeyMiddleware(app)

    async def receive():
        return {}

    async def send(message):
        pass

    scope = {
        "type": "lifespan",
    }

    await middleware(scope, receive, send)

    assert app.called is True
