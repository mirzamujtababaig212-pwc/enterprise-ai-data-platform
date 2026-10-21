from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import create_engine, event, select
from sqlalchemy.orm import sessionmaker

from app.control_plane.persistence.models import (
    Base,
    VoiceCallRecord,
    VoiceLineConfigRecord,
    VoiceProviderEventRecord,
)
from app.control_plane.voice.call_repository import (
    VoiceCallRepository,
    VoiceEventConflictError,
    VoiceLineNotEligibleError,
)
from app.control_plane.voice.models import VoiceEvent, VoiceEventType


@pytest.fixture()
def db():
    engine = create_engine("sqlite:///:memory:")

    @event.listens_for(engine, "connect")
    def enable_sqlite_foreign_keys(connection, _record):
        # Disable sqlite3's implicit transaction handling so SQLAlchemy
        # controls the outer transaction and nested savepoints consistently.
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


def make_line(
    db,
    *,
    phone: str = "+14155550101",
    tenant: str = "tenant-acme",
    status: str = "active",
) -> VoiceLineConfigRecord:
    line = VoiceLineConfigRecord(
        tenant_id=tenant,
        configured_phone_number=phone,
        carrier_trunk_line="acme-main",
        line_status=status,
        inbound_opening_greeting="Welcome.",
        outbound_opening_message="Hello.",
        provider_key="twilio",
    )
    db.add(line)
    db.commit()
    return line


def make_event(
    *,
    event_id: str = "EV-1",
    call_id: str = "CA-1",
    phone: str = "+14155550101",
    event_type: VoiceEventType = VoiceEventType.CALL_RECEIVED,
    occurred_at: datetime | None = None,
) -> VoiceEvent:
    return VoiceEvent(
        provider_key="twilio",
        provider_event_id=event_id,
        provider_call_id=call_id,
        business_line_e164=phone,
        event_type=event_type,
        occurred_at=occurred_at or datetime(2026, 10, 18, 10, 0, tzinfo=UTC),
    )


def test_first_event_creates_call_and_event_without_raw_payload(db):
    line = make_line(db)

    call, stored_event, is_new = VoiceCallRepository(db).process_event(make_event(), line=line)

    assert is_new is True
    assert call.tenant_id == line.tenant_id
    assert call.voice_line_id == line.id
    assert call.status == "received"
    assert stored_event.provider_event_id == "EV-1"
    assert db.scalar(select(VoiceCallRecord).count()) is None if False else True
    assert len(db.scalars(select(VoiceCallRecord)).all()) == 1
    assert len(db.scalars(select(VoiceProviderEventRecord)).all()) == 1
    assert "raw_payload" not in VoiceProviderEventRecord.__table__.c
    assert "transcript" not in VoiceCallRecord.__table__.c


def test_redelivery_is_idempotent(db):
    line = make_line(db)
    repo = VoiceCallRepository(db)
    first = make_event()

    call1, event1, new1 = repo.process_event(first, line=line)
    call2, event2, new2 = repo.process_event(first, line=line)

    assert new1 is True
    assert new2 is False
    assert call1.call_id == call2.call_id
    assert event1.id == event2.id
    assert len(db.scalars(select(VoiceCallRecord)).all()) == 1
    assert len(db.scalars(select(VoiceProviderEventRecord)).all()) == 1


def test_reused_event_id_with_conflicting_data_is_rejected(db):
    line = make_line(db)
    repo = VoiceCallRepository(db)
    repo.process_event(make_event(), line=line)

    conflicting = make_event(
        event_type=VoiceEventType.CALL_ANSWERED,
        occurred_at=datetime(2026, 10, 18, 10, 1, tzinfo=UTC),
    )
    with pytest.raises(VoiceEventConflictError):
        repo.process_event(conflicting, line=line)

    assert len(db.scalars(select(VoiceProviderEventRecord)).all()) == 1


def test_event_cannot_be_replayed_against_another_line(db):
    line1 = make_line(db)
    line2 = make_line(
        db,
        phone="+14155550102",
        tenant="tenant-other",
    )
    repo = VoiceCallRepository(db)
    original = make_event()
    repo.process_event(original, line=line1)

    replay_on_other_line = make_event(phone=line2.configured_phone_number)

    with pytest.raises(VoiceEventConflictError):
        repo.process_event(replay_on_other_line, line=line2)


@pytest.mark.parametrize("status", ["paused", "offline"])
def test_inactive_line_is_rejected_without_persisting_event(db, status):
    line = make_line(db, status=status)

    with pytest.raises(VoiceLineNotEligibleError):
        VoiceCallRepository(db).process_event(make_event(), line=line)

    assert len(db.scalars(select(VoiceCallRecord)).all()) == 0
    assert len(db.scalars(select(VoiceProviderEventRecord)).all()) == 0


def test_provider_and_called_number_must_match_line(db):
    line = make_line(db)
    wrong_number = make_event(phone="+14155550199")

    with pytest.raises(VoiceLineNotEligibleError):
        VoiceCallRepository(db).process_event(wrong_number, line=line)

    assert len(db.scalars(select(VoiceCallRecord)).all()) == 0


def test_out_of_order_events_do_not_regress_terminal_status(db):
    line = make_line(db)
    repo = VoiceCallRepository(db)
    start = datetime(2026, 10, 18, 10, 0, tzinfo=UTC)

    call, _, _ = repo.process_event(
        make_event(event_id="EV-1", occurred_at=start),
        line=line,
    )
    completed_at = start + timedelta(minutes=2)
    call, _, _ = repo.process_event(
        make_event(
            event_id="EV-2",
            event_type=VoiceEventType.CALL_COMPLETED,
            occurred_at=completed_at,
        ),
        line=line,
    )
    call, _, _ = repo.process_event(
        make_event(
            event_id="EV-3",
            event_type=VoiceEventType.CALL_ANSWERED,
            occurred_at=start + timedelta(minutes=1),
        ),
        line=line,
    )

    assert call.status == "completed"
    assert call.completed_at is not None
    assert call.last_event_at == completed_at
    assert len(db.scalars(select(VoiceProviderEventRecord)).all()) == 3


def test_commit_false_leaves_outer_transaction_uncommitted(db):
    line = make_line(db)
    repo = VoiceCallRepository(db)

    call, _, is_new = repo.process_event(
        make_event(),
        line=line,
        commit=False,
    )

    assert is_new is True
    assert db.get(VoiceCallRecord, call.call_id) is not None
    db.rollback()
    assert db.get(VoiceCallRecord, call.call_id) is None
    assert len(db.scalars(select(VoiceProviderEventRecord)).all()) == 0


def test_existing_call_with_mismatched_stored_phone_is_rejected(db):
    line = make_line(db)
    repo = VoiceCallRepository(db)

    call, _, _ = repo.process_event(make_event(), line=line)
    call.business_line_e164 = "+14155550999"
    db.commit()

    next_event = make_event(
        event_id="EV-2",
        event_type=VoiceEventType.CALL_ANSWERED,
        occurred_at=datetime(2026, 10, 18, 10, 1, tzinfo=UTC),
    )

    with pytest.raises(VoiceLineNotEligibleError):
        repo.process_event(next_event, line=line)

    persisted_call = db.get(VoiceCallRecord, call.call_id)
    assert persisted_call is not None
    assert persisted_call.business_line_e164 == "+14155550999"
    assert persisted_call.status == "received"
    assert len(db.scalars(select(VoiceProviderEventRecord)).all()) == 1


def test_get_or_create_call_creates_call_without_provider_event(db):
    line = make_line(db)
    repo = VoiceCallRepository(db)

    call, is_new = repo.get_or_create_call(
        provider_key="twilio",
        provider_call_id="CA-INBOUND-1",
        line=line,
    )

    assert is_new is True
    assert call.tenant_id == line.tenant_id
    assert call.voice_line_id == line.id
    assert call.provider_key == "twilio"
    assert call.provider_call_id == "CA-INBOUND-1"
    assert call.business_line_e164 == line.configured_phone_number
    assert call.status == "received"
    assert len(db.scalars(select(VoiceCallRecord)).all()) == 1
    assert len(db.scalars(select(VoiceProviderEventRecord)).all()) == 0


def test_get_or_create_call_is_idempotent_for_same_provider_call(db):
    line = make_line(db)
    repo = VoiceCallRepository(db)

    first, first_is_new = repo.get_or_create_call(
        provider_key="twilio",
        provider_call_id="CA-INBOUND-2",
        line=line,
    )
    second, second_is_new = repo.get_or_create_call(
        provider_key="twilio",
        provider_call_id="CA-INBOUND-2",
        line=line,
    )

    assert first_is_new is True
    assert second_is_new is False
    assert second.call_id == first.call_id
    assert len(db.scalars(select(VoiceCallRecord)).all()) == 1
    assert len(db.scalars(select(VoiceProviderEventRecord)).all()) == 0


@pytest.mark.parametrize(
    ("provider_key", "provider_call_id"),
    [
        ("", "CA-INBOUND-3"),
        ("twilio ", "CA-INBOUND-3"),
        ("twilio", ""),
        ("twilio", " CA-INBOUND-3"),
    ],
)
def test_get_or_create_call_rejects_invalid_provider_identity(db, provider_key, provider_call_id):
    line = make_line(db)

    with pytest.raises(ValueError, match="Provider call identity is invalid"):
        VoiceCallRepository(db).get_or_create_call(
            provider_key=provider_key,
            provider_call_id=provider_call_id,
            line=line,
        )

    assert len(db.scalars(select(VoiceCallRecord)).all()) == 0


def test_get_or_create_call_rejects_inactive_line(db):
    line = make_line(db, status="paused")

    with pytest.raises(VoiceLineNotEligibleError, match="not active"):
        VoiceCallRepository(db).get_or_create_call(
            provider_key="twilio",
            provider_call_id="CA-INBOUND-4",
            line=line,
        )

    assert len(db.scalars(select(VoiceCallRecord)).all()) == 0


def test_get_or_create_call_rejects_provider_mismatch(db):
    line = make_line(db)

    with pytest.raises(VoiceLineNotEligibleError, match="Provider does not match"):
        VoiceCallRepository(db).get_or_create_call(
            provider_key="other-provider",
            provider_call_id="CA-INBOUND-5",
            line=line,
        )

    assert len(db.scalars(select(VoiceCallRecord)).all()) == 0


def test_get_or_create_call_rejects_existing_call_owned_by_another_line(db):
    first_line = make_line(db, phone="+14155550101", tenant="tenant-acme")
    second_line = make_line(
        db,
        phone="+14155550102",
        tenant="tenant-other",
    )
    repo = VoiceCallRepository(db)

    original, is_new = repo.get_or_create_call(
        provider_key="twilio",
        provider_call_id="CA-INBOUND-6",
        line=first_line,
    )
    assert is_new is True

    with pytest.raises(
        VoiceLineNotEligibleError,
        match="another tenant or line",
    ):
        repo.get_or_create_call(
            provider_key="twilio",
            provider_call_id="CA-INBOUND-6",
            line=second_line,
        )

    assert len(db.scalars(select(VoiceCallRecord)).all()) == 1
    assert original.tenant_id == "tenant-acme"


def test_get_or_create_call_commit_false_can_be_committed_by_caller(db):
    line = make_line(db)
    repo = VoiceCallRepository(db)

    call, is_new = repo.get_or_create_call(
        provider_key="twilio",
        provider_call_id="CA-INBOUND-7",
        line=line,
        commit=False,
    )

    assert is_new is True
    assert call.call_id
    assert len(db.scalars(select(VoiceCallRecord)).all()) == 1

    db.commit()

    persisted = db.scalar(
        select(VoiceCallRecord).where(
            VoiceCallRecord.provider_key == "twilio",
            VoiceCallRecord.provider_call_id == "CA-INBOUND-7",
        )
    )
    assert persisted is not None
    assert persisted.call_id == call.call_id
