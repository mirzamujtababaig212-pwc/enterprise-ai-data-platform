from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime
import uuid
import time
import logging
from ai_platform.llm_gateway.auth.auth import APIKeyMiddleware
from ai_platform.llm_gateway.metrics.metrics import Metrics
from ai_platform.llm_gateway.routing.router import Router
from ai_platform.llm_gateway.registry.provider_registry import registry

app = FastAPI()
app.add_middleware(APIKeyMiddleware)
router = Router()


# Define request/response models
class ChatRequest(BaseModel):
    message: str
    user_id: str | None = None
    model: str | None = "openai"


class ChatResponse(BaseModel):
    reply: str
    metrics: Metrics


class EmbeddingRequest(BaseModel):
    text: str
    model: str | None = "openai"


class EmbeddingResponse(BaseModel):
    vector: list[float]
    metrics: Metrics


class HealthResponse(BaseModel):
    status: str = "ok"
    providers: dict


# Configure logging
logging.basicConfig(
    filename="gateway.log", level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
)


# Endpoints
@app.post("/v1/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    start = time.time()
    reply = router.route_chat(request.dict())

    metrics = Metrics(
        request_id=str(uuid.uuid4()),
        timestamp=datetime.utcnow(),
        latency_ms=int((time.time() - start) * 1000),
        tokens_in=len(request.message.split()),
        tokens_out=len(str(reply).split()),
        estimated_cost=0.001,  # placeholder
        status="success",
    )

    # Log metrics
    logging.info(f"Chat metrics: {metrics.json()}")

    # You could log or return metrics alongside the response
    return ChatResponse(reply=reply, metrics=metrics)


@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def embeddings_endpoint(request: EmbeddingRequest):
    start = time.time()
    vector = router.route_embeddings(request.dict())

    metrics = Metrics(
        request_id=str(uuid.uuid4()),
        timestamp=datetime.utcnow(),
        latency_ms=int((time.time() - start) * 1000),
        tokens_in=len(request.text.split()),
        tokens_out=len(vector),
        estimated_cost=0.002,  # placeholder
        status="success",
    )

    return EmbeddingResponse(vector=vector, metrics=metrics)


@app.get("/v1/models")
async def list_models():
    return {"models": registry.list_providers()}


@app.get("/v1/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(status="ok", providers=router.route_health())
