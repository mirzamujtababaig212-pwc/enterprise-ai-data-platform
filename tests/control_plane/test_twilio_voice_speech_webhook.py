from __future__ import annotations

import logging

from urllib.parse import urlencode

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from starlette.datastructures import FormData
from twilio.request_validator import RequestValidator

from ai_platform.llm_gateway.config.settings import settings
from app.control_plane.persistence.database import get_db
from app.control_plane.persistence.models import (
    VoiceCallRecord,
    VoiceCallTurnRecord,
    VoiceLineConfigRecord,
)
from app.control_plane.routes.twilio_voice_inbound_webhook import (
    SPEECH_PATH,
    router,
)
from app.control_plane.voice.call_repository import VoiceCallRepository
from app.control_plane.voice.turn_repository import VoiceTurnRepository


AUTH_TOKEN = "test-twilio-auth-token"
PHONE = "+14155550101"
CALL_SID = "CA123456789"
TURN_TOKEN = "test-turn-callback-token"
SPEECH_BASE_URL = f"https://voice.example.test{SPEECH_PATH}?source=prod"
SPEECH_CALLBACK_URL = f"{SPEECH_BASE_URL}&turn_token={TURN_TOKEN}"


def speech_items(**overrides: str) -> list[tuple[str, str]]:
    values = {
        "CallSid": CALL_SID,
        "To": PHONE,
        "From": "+14155550199",
        "SpeechResult": "Please schedule an appointment.",
    }
    values.update(overrides)
    return list(values.items())


def signed_headers(
    items: list[tuple[str, str]],
    *,
    url: str = SPEECH_CALLBACK_URL,
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
def speech_env(monkeypatch):
    monkeypatch.setattr(settings, "TWILIO_AUTH_TOKEN", AUTH_TOKEN)
    monkeypatch.setattr(
        settings,
        "VOICE_FINGERPRINT_SECRET",
        "test-voice-fingerprint-secret-at-least-32-bytes",
    )
    monkeypatch.setattr(
        settings,
        "TWILIO_SPEECH_WEBHOOK_URL",
        SPEECH_BASE_URL,
    )

    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    VoiceLineConfigRecord.__table__.create(engine)
    VoiceCallRecord.__table__.create(engine)
    VoiceCallTurnRecord.__table__.create(engine)

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


def seed_turn(session_factory) -> str:
    with session_factory() as session:
        line = VoiceLineConfigRecord(
            tenant_id="tenant-trusted",
            configured_phone_number=PHONE,
            carrier_trunk_line="test-trunk",
            inbound_opening_greeting="Welcome.",
            outbound_opening_message="Hello.",
            line_status="active",
            provider_key="twilio",
        )
        session.add(line)
        session.commit()

        call, created = VoiceCallRepository(session).get_or_create_call(
            provider_key="twilio",
            provider_call_id=CALL_SID,
            line=line,
        )
        assert created is True

        turn, created = VoiceTurnRepository(session).create_or_get(
            tenant_id=line.tenant_id,
            call_id=call.call_id,
            turn_number=1,
            callback_token=TURN_TOKEN,
        )
        assert created is True
        return turn.turn_id


def send_speech(
    client: TestClient,
    *,
    items: list[tuple[str, str]] | None = None,
    signature_override: str | None = None,
    path: str | None = None,
):
    items = items if items is not None else speech_items()
    body = urlencode(items).encode("utf-8")

    return client.post(
        path or f"{SPEECH_PATH}?source=prod&turn_token={TURN_TOKEN}",
        content=body,
        headers=signed_headers(
            items,
            signature_override=signature_override,
        ),
    )


def read_turn(session_factory) -> VoiceCallTurnRecord:
    with session_factory() as session:
        turn = session.scalar(select(VoiceCallTurnRecord))
        assert turn is not None
        session.expunge(turn)
        return turn


def test_valid_speech_is_claimed_and_transcript_is_not_persisted(speech_env):
    client, session_factory = speech_env
    seed_turn(session_factory)

    response = send_speech(client)

    assert response.status_code == 200
    assert "<Response>" in response.text
    assert "can't process that request" in response.text

    turn = read_turn(session_factory)
    assert turn.status == "failed"
    assert turn.failure_code == "agent_not_configured"
    assert turn.request_fingerprint is None
    assert turn.claim_token is not None
    assert "Please schedule an appointment." not in str(turn.__dict__)
    assert not hasattr(turn, "transcript")
    assert not hasattr(turn, "raw_payload")


def test_speech_callback_does_not_log_transcript_or_callback_token(speech_env, caplog):
    client, session_factory = speech_env
    seed_turn(session_factory)
    transcript = "privacy-canary-transcript-7f3a9c"
    caplog.set_level(logging.DEBUG)
    # HTTPX logs the test client's full request URL at INFO, including
    # the callback token. Keep application DEBUG logs captured without
    # treating the test client's own request log as an application log.
    caplog.set_level(logging.WARNING, logger="httpx")

    response = send_speech(
        client,
        items=speech_items(SpeechResult=transcript),
    )

    assert response.status_code == 200
    assert transcript not in response.text
    assert TURN_TOKEN not in response.text
    assert transcript not in caplog.text
    assert TURN_TOKEN not in caplog.text

    turn = read_turn(session_factory)
    assert turn.status == "failed"
    assert turn.failure_code == "agent_not_configured"
    assert turn.request_fingerprint is None
    assert turn.response_text is None
    assert transcript not in str(turn.__dict__)


def test_duplicate_speech_callback_does_not_reclaim_turn(speech_env):
    client, session_factory = speech_env
    seed_turn(session_factory)

    first = send_speech(client)
    second = send_speech(client)

    assert first.status_code == 200
    assert second.status_code == 200
    assert "already been processed" in second.text

    turn = read_turn(session_factory)
    assert turn.status == "failed"
    assert turn.failure_code == "agent_not_configured"


def test_empty_speech_marks_turn_no_input(speech_env):
    client, session_factory = speech_env
    seed_turn(session_factory)

    response = send_speech(
        client,
        items=speech_items(SpeechResult="   "),
    )

    assert response.status_code == 200
    assert "did not catch that" in response.text

    turn = read_turn(session_factory)
    assert turn.status == "no_input"
    assert turn.request_fingerprint is None
    assert turn.failure_code is None


def test_invalid_signature_cannot_change_turn(speech_env):
    client, session_factory = speech_env
    seed_turn(session_factory)

    response = send_speech(
        client,
        signature_override="invalid-signature",
    )

    assert response.status_code == 403
    turn = read_turn(session_factory)
    assert turn.status == "awaiting_input"
    assert turn.request_fingerprint is None


def test_call_identity_mismatch_is_rejected(speech_env):
    client, session_factory = speech_env
    seed_turn(session_factory)

    items = speech_items(CallSid="CA999999999")
    response = send_speech(client, items=items)

    assert response.status_code == 403
    turn = read_turn(session_factory)
    assert turn.status == "awaiting_input"
    assert turn.request_fingerprint is None


def test_callback_query_must_match_configured_base(speech_env):
    client, session_factory = speech_env
    seed_turn(session_factory)

    response = send_speech(
        client,
        path=f"{SPEECH_PATH}?source=other&turn_token={TURN_TOKEN}",
    )

    assert response.status_code == 403
    turn = read_turn(session_factory)
    assert turn.status == "awaiting_input"
