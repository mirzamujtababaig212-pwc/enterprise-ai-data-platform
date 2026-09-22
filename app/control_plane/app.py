from contextlib import asynccontextmanager

from fastapi import FastAPI

from ai_platform.llm_gateway.api.main import app as llm_gateway_app
from ai_platform.llm_gateway.middleware.request_id import RequestIDMiddleware
from app.control_plane.auth import ControlPlaneAPIKeyMiddleware
from app.control_plane.routes.agents import router as agents_router
from app.control_plane.routes.evaluation import router as evaluation_router
from app.control_plane.routes.health import router as health_router
from app.control_plane.routes.llm import router as llm_router
from app.control_plane.routes.ml import router as ml_router
from app.control_plane.routes.platform import router as platform_router
from app.control_plane.routes.rag import router as rag_router
from app.control_plane.dependencies import (
    close_mcp_servers,
    close_rag_vector_store,
    initialize_agents,
    initialize_mcp_servers,
)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    try:
        await initialize_mcp_servers()
        await initialize_agents()
        yield
    finally:
        await close_mcp_servers()
        await close_rag_vector_store()


app = FastAPI(
    title="Enterprise AI Platform Control Plane",
    version="1.0.0",
    description=("Unified API and orchestration boundary for the Enterprise AI Platform."),
    lifespan=lifespan,
)

app.add_middleware(ControlPlaneAPIKeyMiddleware)
app.add_middleware(RequestIDMiddleware)

app.include_router(health_router)
app.include_router(agents_router)
app.include_router(evaluation_router)
app.include_router(platform_router)
app.include_router(llm_router)
app.include_router(ml_router)
app.include_router(rag_router)

app.mount("/", llm_gateway_app)
