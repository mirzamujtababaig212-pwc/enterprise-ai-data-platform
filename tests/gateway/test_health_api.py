from fastapi import FastAPI
from fastapi.testclient import TestClient

from ai_platform.llm_gateway.api.health import router

app = FastAPI()
app.include_router(router)

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
    }
