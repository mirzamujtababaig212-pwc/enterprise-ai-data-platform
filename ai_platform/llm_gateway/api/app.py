"""ASGI application entrypoint."""

from ai_platform.llm_gateway.api.main import app

__all__ = ["app"]
