from __future__ import annotations

import logging

from fastapi import FastAPI

from app.api.chat import router as chat_router
from app.config.settings import Settings
from app.gateway.router import ProviderRouter
from app.providers.openai import OpenAIProvider

settings = Settings.from_environment()

logging.basicConfig(
    level=getattr(
        logging,
        settings.log_level,
        logging.INFO,
    ),
)

providers = {}

openai_credentials = settings.provider_credentials.get("openai", {})

openai_api_key = openai_credentials.get("api_key")

if openai_api_key:
    providers["openai"] = OpenAIProvider(
        api_key=openai_api_key,
        default_model=openai_credentials.get(
            "model",
            "gpt-4o-mini",
        ),
    )


provider_router = ProviderRouter(
    providers=providers,
    default_provider=settings.default_provider,
)


app = FastAPI(
    title="Enterprise AI Platform LLM Gateway",
    version="0.1.0",
)


@app.get("/healthz")
async def healthz():
    return {
        "status": "ok",
        "environment": settings.environment,
    }


@app.get("/readyz")
async def readyz():

    configured_providers = sorted(providers.keys())

    return {
        "status": "ready" if configured_providers else "degraded",
        "default_provider": settings.default_provider,
        "providers": configured_providers,
    }


app.include_router(chat_router)
