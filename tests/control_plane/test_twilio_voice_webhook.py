from __future__ import annotations

from urllib.parse import urlencode

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from starlette.datastructures import FormData
from twilio.request_validator import RequestValidator

from ai_platform.llm_gateway.config.settings import settings
from app.control_plane.persistence.database import get_db
from app.control_plane.persistence.models import (
    VoiceCallRecord,
    VoiceLineConfigRecord,
    VoiceProviderEventRecord,
)
from app.control_plane.routes.twilio_voice_webhook import router


AUTH_TOKEN = "test-twilio-auth-token"
CALLBACK_URL = "https://voice.example.test/api/v1/voice/webhooks/twilio/status?source=prod"
CALLBACK_PATH = "/api/v1/voice/webhooks/twilio/status"
TIMESTAMP = "2026-10-18T10:00:00Z"
PHONE = "+14155550101"


def form_items(**overrides: str) -> list[tuple[str, str]]:
    values = {
        "CallSid": "CA123456789",
        "CallStatus": "in-progress",
        "To": PHONE,
        "From": "+14155550199",
        "SequenceNumber": "2",
        "Timestamp": TIMESTAMP,
    }
    values.update(overrides)
    return list(values.items())


def signed_headers(
    items: list[tuple[str, str]],
    *,
    url: str = CALLBACK_URL,
    signature_override: str | None = None,
) -> dict[str, str]:
    signature = RequestValidator(AUTH_TOKEN).compute_signature(
        url,
        FormData(items),
    )
    if signature_override is not None:
        signature = signature_override

    return {
        "Content-Type": "application/x-www-form-urlencoded",
        "X-Twilio-Signature": signature,
    }


@pytest.fixture
def webhook_env(monkeypatch):
    monkeypatch.setattr(settings, "TWILIO_AUTH_TOKEN", AUTH_TOKEN)
    monkeypatch.setattr(settings, "TWILIO_WEBHOOK_URL", CALLBACK_URL)

    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    # Create only the three voice tables, in foreign-key dependency order.
    VoiceLineConfigRecord.__table__.create(engine)
    VoiceCallRecord.__table__.create(engine)
    VoiceProviderEventRecord.__table__.create(engine)

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
    app.include_router(router)
    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app) as client:
        yield client, session_factory

    app.dependency_overrides.clear()
    engine.dispose()


def add_line(session_factory, *, status: str = "active") -> int:
    with session_factory() as session:
        line = VoiceLineConfigRecord(
            tenant_id="tenant-trusted",
            configured_phone_number=PHONE,
            carrier_trunk_line="test-trunk",
            inbound_opening_greeting="Welcome.",
            outbound_opening_message="Hello.",
            line_status=status,
            provider_key="twilio",
        )
        session.add(line)
        session.commit()
        return line.id


def send_callback(
    client: TestClient,
    items: list[tuple[str, str]] | None = None,
    *,
    path: str | None = None,
    signature_override: str | None = None,
    body: bytes | None = None,
):
    items = items if items is not None else form_items()
    if body is None:
        body = urlencode(items).encode("utf-8")

    return client.post(
        path or f"{CALLBACK_PATH}?source=prod",
        content=body,
        headers=signed_headers(
            items,
            signature_override=signature_override,
        ),
    )


def count_rows(session_factory, model) -> int:
    with session_factory() as session:
        return session.scalar(select(func.count()).select_from(model))


def test_valid_callback_persists_call_and_event_for_trusted_tenant(webhook_env):
    client, session_factory = webhook_env
    line_id = add_line(session_factory)

    response = send_callback(client)

    assert response.status_code == 204
    assert count_rows(session_factory, VoiceCallRecord) == 1
    assert count_rows(session_factory, VoiceProviderEventRecord) == 1

    with session_factory() as session:
        call = session.scalar(select(VoiceCallRecord))
        event = session.scalar(select(VoiceProviderEventRecord))

        assert call is not None
        assert call.tenant_id == "tenant-trusted"
        assert call.voice_line_id == line_id
        assert call.provider_key == "twilio"
        assert call.provider_call_id == "CA123456789"
        assert call.business_line_e164 == PHONE
        assert call.status == "in_progress"

        assert event is not None
        assert event.tenant_id == "tenant-trusted"
        assert event.provider_event_id == "CA123456789:2"
        assert event.event_type == "call_answered"

        # The schema persists event metadata, not the original webhook body.
        assert not hasattr(event, "raw_payload")
        assert not hasattr(event, "transcript")


def test_duplicate_callback_is_idempotent(webhook_env):
    client, session_factory = webhook_env
    add_line(session_factory)

    first = send_callback(client)
    second = send_callback(client)

    assert first.status_code == 204
    assert second.status_code == 204
    assert count_rows(session_factory, VoiceCallRecord) == 1
    assert count_rows(session_factory, VoiceProviderEventRecord) == 1


def test_invalid_signature_is_rejected_without_persisting(webhook_env):
    client, session_factory = webhook_env
    add_line(session_factory)

    response = send_callback(
        client,
        signature_override="invalid-signature",
    )

    assert response.status_code == 403
    assert count_rows(session_factory, VoiceCallRecord) == 0
    assert count_rows(session_factory, VoiceProviderEventRecord) == 0


def test_missing_configuration_fails_closed(webhook_env, monkeypatch):
    client, session_factory = webhook_env
    add_line(session_factory)
    monkeypatch.setattr(settings, "TWILIO_AUTH_TOKEN", "")

    response = send_callback(client)

    assert response.status_code == 503
    assert count_rows(session_factory, VoiceCallRecord) == 0


def test_configured_query_must_match_incoming_query(webhook_env):
    client, session_factory = webhook_env
    add_line(session_factory)

    response = send_callback(
        client,
        path=f"{CALLBACK_PATH}?source=other",
    )

    assert response.status_code == 403
    assert count_rows(session_factory, VoiceCallRecord) == 0


def test_callback_path_must_match_exactly(webhook_env):
    client, session_factory = webhook_env
    add_line(session_factory)

    response = send_callback(
        client,
        path=f"{CALLBACK_PATH}/",
    )

    assert response.status_code in {403, 404, 405}
    assert count_rows(session_factory, VoiceCallRecord) == 0


def test_unregistered_phone_number_is_rejected(webhook_env):
    client, session_factory = webhook_env

    response = send_callback(
        client,
        form_items(To="+14155550999"),
    )

    assert response.status_code == 404
    assert count_rows(session_factory, VoiceCallRecord) == 0


def test_paused_line_cannot_accept_callback(webhook_env):
    client, session_factory = webhook_env
    add_line(session_factory, status="paused")

    response = send_callback(client)

    assert response.status_code == 403
    assert count_rows(session_factory, VoiceCallRecord) == 0
    assert count_rows(session_factory, VoiceProviderEventRecord) == 0


def test_line_configured_for_another_provider_is_rejected(webhook_env):
    client, session_factory = webhook_env

    with session_factory() as session:
        line = VoiceLineConfigRecord(
            tenant_id="tenant-trusted",
            configured_phone_number=PHONE,
            carrier_trunk_line="test-trunk",
            inbound_opening_greeting="Welcome.",
            outbound_opening_message="Hello.",
            line_status="active",
            provider_key="other-provider",
        )
        session.add(line)
        session.commit()

    response = send_callback(client)

    assert response.status_code == 403
    assert count_rows(session_factory, VoiceCallRecord) == 0


def test_oversized_body_is_rejected_before_signature_validation(webhook_env):
    client, session_factory = webhook_env
    oversized = b"x" * (64 * 1024 + 1)

    response = send_callback(client, body=oversized)

    assert response.status_code == 413
    assert count_rows(session_factory, VoiceCallRecord) == 0


def test_completed_callback_updates_call_state(webhook_env):
    client, session_factory = webhook_env
    add_line(session_factory)

    initial = send_callback(client)
    completed_items = form_items(
        CallStatus="completed",
        SequenceNumber="3",
        Timestamp="2026-10-18T10:05:00Z",
    )
    completed = send_callback(client, completed_items)

    assert initial.status_code == 204
    assert completed.status_code == 204
    assert count_rows(session_factory, VoiceCallRecord) == 1
    assert count_rows(session_factory, VoiceProviderEventRecord) == 2

    with session_factory() as session:
        call = session.scalar(select(VoiceCallRecord))
        assert call is not None
        assert call.status == "completed"
        assert call.completed_at is not None
