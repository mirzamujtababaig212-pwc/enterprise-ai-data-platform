from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.control_plane.persistence.database import get_db
from app.control_plane.persistence.models import VoiceLineConfigRecord
from app.control_plane.routes.voice import router


@pytest.fixture()
def client():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    VoiceLineConfigRecord.__table__.create(engine)
    session_factory = sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
    )

    def override_get_db():
        session = session_factory()
        try:
            yield session
        finally:
            session.close()

    app = FastAPI()

    @app.middleware("http")
    async def test_identity(request, call_next):
        request.state.tenant_id = "tenant-acme"
        request.state.principal = "test-principal"
        return await call_next(request)

    app.include_router(router)
    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()
    engine.dispose()


def create_payload(phone="+14155550101"):
    return {
        "configured_phone_number": phone,
        "carrier_trunk_line": "acme-main",
        "inbound_opening_greeting": "Thanks for calling Acme.",
        "outbound_opening_message": "Hello from Acme.",
    }


def test_create_voice_line_starts_paused(client):
    response = client.post("/api/v1/voice/lines", json=create_payload())

    assert response.status_code == 201
    body = response.json()
    assert body["tenant_id"] == "tenant-acme"
    assert body["line_status"] == "paused"
    assert body["provider_sync_status"] == "not_configured"
    assert body["voice_synthesizer"] == "polly-kajal"
    assert body["spoken_language"] == "hi-IN"


def test_client_cannot_set_tenant_or_activation_status(client):
    response = client.post(
        "/api/v1/voice/lines",
        json={**create_payload(), "tenant_id": "tenant-other"},
    )
    assert response.status_code == 422

    response = client.post(
        "/api/v1/voice/lines",
        json={**create_payload(), "line_status": "active"},
    )
    assert response.status_code == 422


def test_create_rejects_invalid_phone_and_voice_settings(client):
    invalid_phone = client.post(
        "/api/v1/voice/lines",
        json=create_payload("555-0101"),
    )
    assert invalid_phone.status_code == 422

    invalid_settings = client.post(
        "/api/v1/voice/lines",
        json={**create_payload(), "barge_in_sensitivity": 99},
    )
    assert invalid_settings.status_code == 422


def test_duplicate_phone_returns_conflict(client):
    first = client.post("/api/v1/voice/lines", json=create_payload())
    second = client.post("/api/v1/voice/lines", json=create_payload())

    assert first.status_code == 201
    assert second.status_code == 409
    assert second.json()["detail"] == "Configured phone number is already assigned."


def test_list_and_get_are_scoped_to_authenticated_tenant(client):
    created = client.post("/api/v1/voice/lines", json=create_payload())
    line_id = created.json()["id"]

    listed = client.get("/api/v1/voice/lines")
    fetched = client.get(f"/api/v1/voice/lines/{line_id}")
    missing = client.get(f"/api/v1/voice/lines/{line_id + 100}")

    assert listed.status_code == 200
    assert len(listed.json()) == 1
    assert fetched.status_code == 200
    assert fetched.json()["id"] == line_id
    assert missing.status_code == 404


def test_patch_updates_allowed_fields(client):
    created = client.post("/api/v1/voice/lines", json=create_payload())
    line_id = created.json()["id"]

    updated = client.patch(
        f"/api/v1/voice/lines/{line_id}",
        json={
            "inbound_opening_greeting": "Welcome to Acme Support.",
            "barge_in_sensitivity": 4,
        },
    )

    assert updated.status_code == 200
    assert updated.json()["inbound_opening_greeting"] == "Welcome to Acme Support."
    assert updated.json()["barge_in_sensitivity"] == 4
    assert updated.json()["line_status"] == "paused"


def test_patch_rejects_empty_unknown_or_protected_changes(client):
    created = client.post("/api/v1/voice/lines", json=create_payload())
    line_id = created.json()["id"]
    url = f"/api/v1/voice/lines/{line_id}"

    assert client.patch(url, json={}).status_code == 422
    assert client.patch(url, json={"tenant_id": "tenant-other"}).status_code == 422
    assert client.patch(url, json={"provider_sync_status": "synced"}).status_code == 422
    assert client.patch(url, json={"line_status": "active"}).status_code == 422
    assert (
        client.patch(
            url,
            json={"inbound_opening_greeting": None},
        ).status_code
        == 422
    )


def test_patch_duplicate_phone_returns_conflict(client):
    client.post("/api/v1/voice/lines", json=create_payload("+14155550101"))
    second = client.post(
        "/api/v1/voice/lines",
        json=create_payload("+14155550102"),
    )
    line_id = second.json()["id"]

    response = client.patch(
        f"/api/v1/voice/lines/{line_id}",
        json={"configured_phone_number": "+14155550101"},
    )

    assert response.status_code == 409


def test_pagination_is_validated(client):
    assert client.get("/api/v1/voice/lines?limit=0").status_code == 422
    assert client.get("/api/v1/voice/lines?limit=501").status_code == 422
    assert client.get("/api/v1/voice/lines?offset=-1").status_code == 422
