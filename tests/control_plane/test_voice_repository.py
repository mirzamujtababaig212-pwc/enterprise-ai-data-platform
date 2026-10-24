from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from app.control_plane.persistence.models import VoiceLineConfigRecord
from app.control_plane.voice.repository import (
    DuplicateVoiceLinePhoneNumberError,
    VoiceLineNotFoundError,
    VoiceLineRepository,
)


@pytest.fixture()
def repository():
    engine = create_engine("sqlite:///:memory:")
    VoiceLineConfigRecord.__table__.create(engine)

    session_factory = sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
    )
    session = session_factory()

    try:
        yield VoiceLineRepository(session)
    finally:
        session.close()
        engine.dispose()


def line_values(
    phone: str = "+14155550101",
    *,
    greeting: str = "Thanks for calling Acme.",
) -> dict[str, object]:
    return {
        "configured_phone_number": phone,
        "carrier_trunk_line": "acme-main",
        "inbound_opening_greeting": greeting,
        "outbound_opening_message": "Hello from Acme.",
    }


def test_create_uses_safe_defaults_and_round_trips(repository) -> None:
    line = repository.create(
        tenant_id="tenant-acme",
        values=line_values(),
    )

    assert line.id is not None
    assert line.tenant_id == "tenant-acme"
    assert line.line_status == "paused"
    assert line.provider_sync_status == "not_configured"
    assert repository.get_for_tenant(line.id, "tenant-acme").id == line.id


def test_get_and_list_are_tenant_scoped(repository) -> None:
    acme = repository.create(
        tenant_id="tenant-acme",
        values=line_values("+14155550101"),
    )
    repository.create(
        tenant_id="tenant-other",
        values=line_values("+14155550102"),
    )

    assert repository.get_for_tenant(acme.id, "tenant-other") is None
    assert [row.tenant_id for row in repository.list_for_tenant("tenant-acme")] == ["tenant-acme"]


def test_list_supports_bounded_pagination(repository) -> None:
    for index in range(3):
        repository.create(
            tenant_id="tenant-acme",
            values=line_values(f"+1415555010{index + 1}"),
        )

    assert len(repository.list_for_tenant("tenant-acme", limit=2)) == 2
    assert len(repository.list_for_tenant("tenant-acme", limit=2, offset=2)) == 1
    with pytest.raises(ValueError):
        repository.list_for_tenant("tenant-acme", limit=0)
    with pytest.raises(ValueError):
        repository.list_for_tenant("tenant-acme", offset=-1)


def test_update_changes_allowed_fields_and_refreshes_timestamp(repository) -> None:
    line = repository.create(
        tenant_id="tenant-acme",
        values=line_values(),
    )
    old_updated_at = line.updated_at
    # Ensure a measurable timestamp difference even with coarse clock precision.
    line.updated_at = datetime.now(UTC) - timedelta(seconds=5)
    repository._session.flush()

    updated = repository.update_for_tenant(
        line.id,
        tenant_id="tenant-acme",
        changes={"inbound_opening_greeting": "Welcome to Acme."},
    )

    assert updated.inbound_opening_greeting == "Welcome to Acme."

    # SQLite drops timezone information when reading DateTime columns.
    # Normalize values to UTC-aware datetimes before comparing them.
    def as_utc(value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)

    assert as_utc(updated.updated_at) > as_utc(old_updated_at)
    assert as_utc(updated.updated_at) > datetime.now(UTC) - timedelta(seconds=2)


def test_update_for_other_tenant_is_not_found(repository) -> None:
    line = repository.create(
        tenant_id="tenant-acme",
        values=line_values(),
    )

    with pytest.raises(VoiceLineNotFoundError):
        repository.update_for_tenant(
            line.id,
            tenant_id="tenant-other",
            changes={"line_status": "active"},
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"tenant_id": "tenant-other"},
        {"id": 99},
        {"created_at": datetime.now(UTC)},
        {"updated_at": datetime.now(UTC)},
        {"not_a_voice_field": "value"},
        {},
    ],
)
def test_update_rejects_protected_unknown_or_empty_changes(
    repository,
    changes: dict[str, object],
) -> None:
    line = repository.create(
        tenant_id="tenant-acme",
        values=line_values(),
    )

    with pytest.raises(ValueError):
        repository.update_for_tenant(
            line.id,
            tenant_id="tenant-acme",
            changes=changes,
        )


def test_create_rejects_tenant_override(repository) -> None:
    with pytest.raises(ValueError, match="Protected"):
        repository.create(
            tenant_id="tenant-acme",
            values={**line_values(), "tenant_id": "tenant-other"},
        )


def test_duplicate_phone_number_raises_safe_domain_error(repository) -> None:
    repository.create(
        tenant_id="tenant-acme",
        values=line_values("+14155550101"),
    )

    with pytest.raises(DuplicateVoiceLinePhoneNumberError):
        repository.create(
            tenant_id="tenant-other",
            values=line_values("+14155550101"),
        )


def test_update_phone_conflict_does_not_leak_other_tenant(repository) -> None:
    repository.create(
        tenant_id="tenant-acme",
        values=line_values("+14155550101"),
    )
    other = repository.create(
        tenant_id="tenant-other",
        values=line_values("+14155550102"),
    )

    with pytest.raises(DuplicateVoiceLinePhoneNumberError):
        repository.update_for_tenant(
            other.id,
            tenant_id="tenant-other",
            changes={"configured_phone_number": "+14155550101"},
        )

    # The failed owned transaction was rolled back.
    persisted = repository._session.scalar(
        select(VoiceLineConfigRecord).where(VoiceLineConfigRecord.id == other.id)
    )
    assert persisted.configured_phone_number == "+14155550102"


def test_commit_false_leaves_transaction_for_caller(repository) -> None:
    line = repository.create(
        tenant_id="tenant-acme",
        values=line_values(),
        commit=False,
    )

    assert line.id is not None
    repository._session.commit()
    assert repository.get_for_tenant(line.id, "tenant-acme") is not None


def test_get_by_phone_number_resolves_authoritative_tenant(repository) -> None:
    line = repository.create(
        tenant_id="tenant-acme",
        values=line_values("+14155550101"),
    )
    repository.create(
        tenant_id="tenant-other",
        values=line_values("+14155550102"),
    )

    resolved = repository.get_by_phone_number("+14155550101")

    assert resolved is not None
    assert resolved.id == line.id
    assert resolved.tenant_id == "tenant-acme"
    assert resolved.configured_phone_number == "+14155550101"


def test_get_by_phone_number_returns_none_for_unknown_number(repository) -> None:
    assert repository.get_by_phone_number("+14155550999") is None


def test_get_by_phone_number_returns_paused_line_for_explicit_eligibility_check(
    repository,
) -> None:
    line = repository.create(
        tenant_id="tenant-acme",
        values=line_values("+14155550101"),
    )

    resolved = repository.get_by_phone_number("+14155550101")

    assert resolved is not None
    assert resolved.id == line.id
    assert resolved.line_status == "paused"


@pytest.mark.parametrize("phone_number", ["", "   ", None])
def test_get_by_phone_number_rejects_empty_input(
    repository,
    phone_number: object,
) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        repository.get_by_phone_number(phone_number)
