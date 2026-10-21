from __future__ import annotations

from datetime import UTC, datetime

import pytest
from sqlalchemy import create_engine, event, select
from sqlalchemy.orm import sessionmaker

from app.control_plane.persistence.models import (
    Base,
    VoiceCallRecord,
    VoiceCallTurnRecord,
    VoiceLineConfigRecord,
    VoiceProviderEventRecord,
)
from app.control_plane.voice.call_repository import VoiceCallRepository
from app.control_plane.voice.models import VoiceEvent, VoiceEventType
from app.control_plane.voice.turn_repository import (
    VoiceTurnConflictError,
    VoiceTurnNotFoundError,
    VoiceTurnRepository,
)


@pytest.fixture()
def db():
    engine = create_engine("sqlite:///:memory:")

    @event.listens_for(engine, "connect")
    def enable_sqlite_foreign_keys(connection, _record):
        connection.isolation_level = None
        cursor = connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    @event.listens_for(engine, "begin")
    def begin_sqlite_transaction(connection):
        connection.exec_driver_sql("BEGIN")

    Base.metadata.create_all(
        engine,
        tables=[
            VoiceLineConfigRecord.__table__,
            VoiceCallRecord.__table__,
            VoiceProviderEventRecord.__table__,
            VoiceCallTurnRecord.__table__,
        ],
    )
    factory = sessionmaker(
        bind=engine,
        autoflush=False,
        expire_on_commit=False,
    )
    session = factory()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()


def make_call(db, *, tenant_id="tenant-acme"):
    line = VoiceLineConfigRecord(
        tenant_id=tenant_id,
        configured_phone_number="+14155550101",
        carrier_trunk_line="acme-main",
        line_status="active",
        inbound_opening_greeting="Welcome.",
        outbound_opening_message="Hello.",
        provider_key="twilio",
    )
    db.add(line)
    db.commit()

    event = VoiceEvent(
        provider_key="twilio",
        provider_event_id="EV-inbound",
        provider_call_id="CA-inbound",
        business_line_e164="+14155550101",
        event_type=VoiceEventType.CALL_RECEIVED,
        occurred_at=datetime(2026, 10, 18, 10, 0, tzinfo=UTC),
    )
    call, _, _ = VoiceCallRepository(db).process_event(event, line=line)
    return call


def make_turn(db):
    call = make_call(db)
    return VoiceTurnRepository(db).create_or_get(
        tenant_id=call.tenant_id,
        call_id=call.call_id,
        turn_number=1,
        callback_token="opaque-token-1",
    )[0]


FINGERPRINT = "a" * 64


def test_create_or_get_is_idempotent(db):
    call = make_call(db)
    repo = VoiceTurnRepository(db)

    first, created1 = repo.create_or_get(
        tenant_id=call.tenant_id,
        call_id=call.call_id,
        turn_number=1,
        callback_token="token-1",
    )
    second, created2 = repo.create_or_get(
        tenant_id=call.tenant_id,
        call_id=call.call_id,
        turn_number=1,
        callback_token="token-1",
    )

    assert created1 is True
    assert created2 is False
    assert first.turn_id == second.turn_id
    assert len(db.scalars(select(VoiceCallTurnRecord)).all()) == 1


def test_turn_number_cannot_be_reused_with_another_token(db):
    turn = make_turn(db)
    repo = VoiceTurnRepository(db)

    with pytest.raises(VoiceTurnConflictError):
        repo.create_or_get(
            tenant_id=turn.tenant_id,
            call_id=turn.call_id,
            turn_number=turn.turn_number,
            callback_token="different-token",
        )


def test_create_rejects_call_from_another_tenant(db):
    call = make_call(db)
    repo = VoiceTurnRepository(db)

    with pytest.raises(VoiceTurnNotFoundError):
        repo.create_or_get(
            tenant_id="another-tenant",
            call_id=call.call_id,
            turn_number=1,
            callback_token="cross-tenant-token",
        )


def test_claim_only_succeeds_once_for_same_fingerprint(db):
    turn = make_turn(db)
    repo = VoiceTurnRepository(db)

    first, claimed1 = repo.claim(
        tenant_id=turn.tenant_id,
        turn_id=turn.turn_id,
        request_fingerprint=FINGERPRINT,
    )
    second, claimed2 = repo.claim(
        tenant_id=turn.tenant_id,
        turn_id=turn.turn_id,
        request_fingerprint=FINGERPRINT,
    )

    assert claimed1 is True
    assert claimed2 is False
    assert first.status == second.status == "processing"
    assert first.request_fingerprint == FINGERPRINT


def test_claim_rejects_different_fingerprint(db):
    turn = make_turn(db)
    repo = VoiceTurnRepository(db)
    repo.claim(
        tenant_id=turn.tenant_id,
        turn_id=turn.turn_id,
        request_fingerprint=FINGERPRINT,
    )

    with pytest.raises(VoiceTurnConflictError):
        repo.claim(
            tenant_id=turn.tenant_id,
            turn_id=turn.turn_id,
            request_fingerprint="b" * 64,
        )


def test_complete_is_idempotent_and_does_not_store_transcript(db):
    turn = make_turn(db)
    repo = VoiceTurnRepository(db)
    repo.claim(
        tenant_id=turn.tenant_id,
        turn_id=turn.turn_id,
        request_fingerprint=FINGERPRINT,
    )

    first, completed1 = repo.complete(
        tenant_id=turn.tenant_id,
        turn_id=turn.turn_id,
        request_fingerprint=FINGERPRINT,
        response_text="Your request is complete.",
        agent_run_id="run-123",
    )
    second, completed2 = repo.complete(
        tenant_id=turn.tenant_id,
        turn_id=turn.turn_id,
        request_fingerprint=FINGERPRINT,
        response_text="Your request is complete.",
        agent_run_id="run-123",
    )

    assert completed1 is True
    assert completed2 is False
    assert first.status == second.status == "completed"
    assert second.response_text == "Your request is complete."
    assert "transcript" not in VoiceCallTurnRecord.__table__.c
    assert "raw_payload" not in VoiceCallTurnRecord.__table__.c


def test_complete_rejects_different_replay_response(db):
    turn = make_turn(db)
    repo = VoiceTurnRepository(db)
    repo.claim(
        tenant_id=turn.tenant_id,
        turn_id=turn.turn_id,
        request_fingerprint=FINGERPRINT,
    )
    repo.complete(
        tenant_id=turn.tenant_id,
        turn_id=turn.turn_id,
        request_fingerprint=FINGERPRINT,
        response_text="First response.",
        agent_run_id="run-123",
    )

    with pytest.raises(VoiceTurnConflictError):
        repo.complete(
            tenant_id=turn.tenant_id,
            turn_id=turn.turn_id,
            request_fingerprint=FINGERPRINT,
            response_text="Conflicting response.",
            agent_run_id="run-123",
        )


def test_failure_is_terminal_and_idempotent(db):
    turn = make_turn(db)
    repo = VoiceTurnRepository(db)
    repo.claim(
        tenant_id=turn.tenant_id,
        turn_id=turn.turn_id,
        request_fingerprint=FINGERPRINT,
    )

    first, failed1 = repo.fail(
        tenant_id=turn.tenant_id,
        turn_id=turn.turn_id,
        request_fingerprint=FINGERPRINT,
        failure_code="agent_execution_failed",
    )
    second, failed2 = repo.fail(
        tenant_id=turn.tenant_id,
        turn_id=turn.turn_id,
        request_fingerprint=FINGERPRINT,
        failure_code="agent_execution_failed",
    )

    assert failed1 is True
    assert failed2 is False
    assert first.status == second.status == "failed"


def test_claim_requires_sha256_fingerprint(db):
    turn = make_turn(db)

    with pytest.raises(ValueError):
        VoiceTurnRepository(db).claim(
            tenant_id=turn.tenant_id,
            turn_id=turn.turn_id,
            request_fingerprint="not-a-sha256",
        )


def test_get_for_call_turn_returns_existing_turn_with_same_identity(db):
    turn = make_turn(db)

    found = VoiceTurnRepository(db).get_for_call_turn(
        tenant_id=turn.tenant_id,
        call_id=turn.call_id,
        turn_number=turn.turn_number,
    )

    assert found is not None
    assert found.turn_id == turn.turn_id
    assert found.callback_token == turn.callback_token


def test_get_for_call_turn_is_tenant_scoped(db):
    turn = make_turn(db)

    found = VoiceTurnRepository(db).get_for_call_turn(
        tenant_id="another-tenant",
        call_id=turn.call_id,
        turn_number=turn.turn_number,
    )

    assert found is None


def test_mark_no_input_is_idempotent_for_same_turn(db):
    turn = make_turn(db)
    repo = VoiceTurnRepository(db)

    first, changed1 = repo.mark_no_input(
        tenant_id=turn.tenant_id,
        turn_id=turn.turn_id,
    )
    second, changed2 = repo.mark_no_input(
        tenant_id=turn.tenant_id,
        turn_id=turn.turn_id,
    )

    assert changed1 is True
    assert changed2 is False
    assert first.status == second.status == "no_input"
    assert second.request_fingerprint is None
    assert second.response_text is None


def test_mark_no_input_rejects_processing_turn(db):
    turn = make_turn(db)
    repo = VoiceTurnRepository(db)
    repo.claim(
        tenant_id=turn.tenant_id,
        turn_id=turn.turn_id,
        request_fingerprint=FINGERPRINT,
    )

    with pytest.raises(VoiceTurnConflictError):
        repo.mark_no_input(tenant_id=turn.tenant_id, turn_id=turn.turn_id)


def test_mark_no_input_rejects_turn_from_another_tenant(db):
    turn = make_turn(db)

    with pytest.raises(VoiceTurnNotFoundError):
        VoiceTurnRepository(db).mark_no_input(
            tenant_id="another-tenant",
            turn_id=turn.turn_id,
        )


def test_claim_transient_does_not_persist_transcript_fingerprint(db):
    turn = make_turn(db)
    repo = VoiceTurnRepository(db)

    claimed_turn, claimed, token = repo.claim_transient(
        tenant_id=turn.tenant_id,
        turn_id=turn.turn_id,
    )

    assert claimed is True
    assert token is not None
    assert claimed_turn.status == "processing"
    assert claimed_turn.claim_token == token
    assert claimed_turn.request_fingerprint is None


def test_claim_transient_only_claims_once(db):
    turn = make_turn(db)
    repo = VoiceTurnRepository(db)

    _, claimed1, token1 = repo.claim_transient(
        tenant_id=turn.tenant_id,
        turn_id=turn.turn_id,
    )
    _, claimed2, token2 = repo.claim_transient(
        tenant_id=turn.tenant_id,
        turn_id=turn.turn_id,
    )

    assert claimed1 is True
    assert token1 is not None
    assert claimed2 is False
    assert token2 is None


def test_complete_transient_completes_claimed_turn(db):
    turn = make_turn(db)
    repo = VoiceTurnRepository(db)

    _, claimed, token = repo.claim_transient(
        tenant_id=turn.tenant_id,
        turn_id=turn.turn_id,
    )
    assert claimed is True
    assert token is not None

    completed, changed = repo.complete_transient(
        tenant_id=turn.tenant_id,
        turn_id=turn.turn_id,
        claim_token=token,
        response_text="Your appointment request has been recorded.",
        agent_run_id="run-123",
    )

    assert changed is True
    assert completed.status == "completed"
    assert completed.response_text == ("Your appointment request has been recorded.")
    assert completed.agent_run_id == "run-123"
    assert completed.failure_code is None
    assert completed.request_fingerprint is None
    assert completed.claim_token == token
    assert completed.completed_at is not None


def test_complete_transient_rejects_wrong_claim_token(db):
    turn = make_turn(db)
    repo = VoiceTurnRepository(db)

    _, claimed, token = repo.claim_transient(
        tenant_id=turn.tenant_id,
        turn_id=turn.turn_id,
    )
    assert claimed is True
    assert token is not None

    with pytest.raises(VoiceTurnConflictError):
        repo.complete_transient(
            tenant_id=turn.tenant_id,
            turn_id=turn.turn_id,
            claim_token="00000000-0000-0000-0000-000000000000",
            response_text="Response",
            agent_run_id=None,
        )

    current = repo.get(tenant_id=turn.tenant_id, turn_id=turn.turn_id)
    assert current.status == "processing"

    completed, changed = repo.complete_transient(
        tenant_id=turn.tenant_id,
        turn_id=turn.turn_id,
        claim_token=token,
        response_text="Response",
        agent_run_id=None,
    )
    assert changed is True
    assert completed.status == "completed"


def test_complete_transient_rejects_replay_after_completion(db):
    turn = make_turn(db)
    repo = VoiceTurnRepository(db)

    _, claimed, token = repo.claim_transient(
        tenant_id=turn.tenant_id,
        turn_id=turn.turn_id,
    )
    assert claimed is True
    assert token is not None

    repo.complete_transient(
        tenant_id=turn.tenant_id,
        turn_id=turn.turn_id,
        claim_token=token,
        response_text="First response",
        agent_run_id=None,
    )

    with pytest.raises(VoiceTurnConflictError):
        repo.complete_transient(
            tenant_id=turn.tenant_id,
            turn_id=turn.turn_id,
            claim_token=token,
            response_text="First response",
            agent_run_id=None,
        )


def test_complete_transient_rejects_another_tenant(db):
    turn = make_turn(db)
    repo = VoiceTurnRepository(db)

    _, claimed, token = repo.claim_transient(
        tenant_id=turn.tenant_id,
        turn_id=turn.turn_id,
    )
    assert claimed is True
    assert token is not None

    with pytest.raises(VoiceTurnNotFoundError):
        repo.complete_transient(
            tenant_id="another-tenant",
            turn_id=turn.turn_id,
            claim_token=token,
            response_text="Response",
            agent_run_id=None,
        )

    current = repo.get(tenant_id=turn.tenant_id, turn_id=turn.turn_id)
    assert current.status == "processing"


@pytest.mark.parametrize(
    "claim_token",
    ["", "   ", None, 123, "x" * 37],
)
def test_complete_transient_rejects_invalid_claim_token(db, claim_token):
    turn = make_turn(db)
    repo = VoiceTurnRepository(db)

    with pytest.raises(ValueError, match="claim_token"):
        repo.complete_transient(
            tenant_id=turn.tenant_id,
            turn_id=turn.turn_id,
            claim_token=claim_token,
            response_text="Response",
            agent_run_id=None,
        )


@pytest.mark.parametrize(
    "agent_run_id",
    ["", "   ", 123, "x" * 37],
)
def test_complete_transient_rejects_invalid_agent_run_id(db, agent_run_id):
    turn = make_turn(db)
    repo = VoiceTurnRepository(db)

    _, claimed, token = repo.claim_transient(
        tenant_id=turn.tenant_id,
        turn_id=turn.turn_id,
    )
    assert claimed is True
    assert token is not None

    with pytest.raises(ValueError, match="agent_run_id"):
        repo.complete_transient(
            tenant_id=turn.tenant_id,
            turn_id=turn.turn_id,
            claim_token=token,
            response_text="Response",
            agent_run_id=agent_run_id,
        )


def test_complete_transient_rejects_oversized_response(db):
    turn = make_turn(db)
    repo = VoiceTurnRepository(db)

    _, claimed, token = repo.claim_transient(
        tenant_id=turn.tenant_id,
        turn_id=turn.turn_id,
    )
    assert claimed is True
    assert token is not None

    with pytest.raises(ValueError, match="8000"):
        repo.complete_transient(
            tenant_id=turn.tenant_id,
            turn_id=turn.turn_id,
            claim_token=token,
            response_text="x" * 8001,
            agent_run_id=None,
        )


def test_fail_transient_requires_current_claim_token(db):
    turn = make_turn(db)
    repo = VoiceTurnRepository(db)

    _, claimed, token = repo.claim_transient(
        tenant_id=turn.tenant_id,
        turn_id=turn.turn_id,
    )
    assert claimed is True
    assert token is not None

    with pytest.raises(VoiceTurnConflictError):
        repo.fail_transient(
            tenant_id=turn.tenant_id,
            turn_id=turn.turn_id,
            claim_token="00000000-0000-0000-0000-000000000000",
            failure_code="agent_not_configured",
        )

    current = repo.get(tenant_id=turn.tenant_id, turn_id=turn.turn_id)
    assert current.status == "processing"

    failed, changed = repo.fail_transient(
        tenant_id=turn.tenant_id,
        turn_id=turn.turn_id,
        claim_token=token,
        failure_code="agent_not_configured",
    )

    assert changed is True
    assert failed.status == "failed"
    assert failed.request_fingerprint is None
    assert failed.claim_token == token


def test_fail_transient_cannot_change_another_tenant_turn(db):
    turn = make_turn(db)
    repo = VoiceTurnRepository(db)

    with pytest.raises(VoiceTurnNotFoundError):
        repo.claim_transient(
            tenant_id="another-tenant",
            turn_id=turn.turn_id,
        )

    assert (
        repo.get(
            tenant_id=turn.tenant_id,
            turn_id=turn.turn_id,
        ).status
        == "awaiting_input"
    )


@pytest.mark.parametrize(
    "claim_token",
    [
        "",
        "   ",
        "x" * 37,
        None,
        123,
    ],
)
def test_fail_transient_rejects_invalid_claim_token(db, claim_token):
    turn = make_turn(db)
    repo = VoiceTurnRepository(db)

    with pytest.raises(ValueError, match="claim_token"):
        repo.fail_transient(
            tenant_id=turn.tenant_id,
            turn_id=turn.turn_id,
            claim_token=claim_token,
            failure_code="agent_not_configured",
        )

    current = repo.get(tenant_id=turn.tenant_id, turn_id=turn.turn_id)
    assert current.status == "awaiting_input"


@pytest.mark.parametrize(
    "failure_code",
    [
        "",
        "   ",
        "x" * 101,
        None,
        123,
    ],
)
def test_fail_transient_rejects_invalid_failure_code(db, failure_code):
    turn = make_turn(db)
    repo = VoiceTurnRepository(db)

    with pytest.raises(ValueError, match="failure_code"):
        repo.fail_transient(
            tenant_id=turn.tenant_id,
            turn_id=turn.turn_id,
            claim_token="valid-claim-token",
            failure_code=failure_code,
        )

    current = repo.get(tenant_id=turn.tenant_id, turn_id=turn.turn_id)
    assert current.status == "awaiting_input"
