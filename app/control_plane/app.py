from fastapi import FastAPI

from ai_platform.llm_gateway.middleware.request_id import RequestIDMiddleware
from app.control_plane.auth import ControlPlaneAPIKeyMiddleware
from app.control_plane.routes.health import router as health_router
from app.control_plane.routes.llm import router as llm_router
from app.control_plane.routes.ml import router as ml_router
from app.control_plane.routes.platform import router as platform_router
from app.control_plane.routes.rag import router as rag_router

app = FastAPI(
    title="Enterprise AI Platform Control Plane",
    version="1.0.0",
    description=("Unified API and orchestration boundary for the Enterprise AI Platform."),
)

app.add_middleware(ControlPlaneAPIKeyMiddleware)
app.add_middleware(RequestIDMiddleware)

app.include_router(health_router)
app.include_router(platform_router)
app.include_router(llm_router)
app.include_router(ml_router)
app.include_router(rag_router)
