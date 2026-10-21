from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.control_plane.persistence.models import (
    VoiceCallRecord,
    VoiceLineConfigRecord,
    VoiceProviderEventRecord,
)
from app.control_plane.voice.lifecycle import next_call_status
from app.control_plane.voice.models import VoiceCallStatus, VoiceEvent


class VoiceEventConflictError(ValueError):
    """A provider event ID was reused with conflicting event data."""


class VoiceLineNotEligibleError(ValueError):
    """The trusted voice line cannot accept this event."""


class VoiceCallRepository:
    """Persist normalized events and call state without raw payloads."""

    def __init__(self, session: Session) -> None:
        self._session = session

    @staticmethod
    def _utc(value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)

    @classmethod
    def _matches_event(
        cls,
        stored: VoiceProviderEventRecord,
        event: VoiceEvent,
    ) -> bool:
        return (
            stored.provider_call_id == event.provider_call_id
            and stored.event_type == event.event_type.value
            and cls._utc(stored.occurred_at) == cls._utc(event.occurred_at)
        )

    def _get_call(self, provider_key: str, provider_call_id: str) -> VoiceCallRecord | None:
        return self._session.scalar(
            select(VoiceCallRecord).where(
                VoiceCallRecord.provider_key == provider_key,
                VoiceCallRecord.provider_call_id == provider_call_id,
            )
        )

    def _get_event(
        self, provider_key: str, provider_event_id: str
    ) -> VoiceProviderEventRecord | None:
        return self._session.scalar(
            select(VoiceProviderEventRecord).where(
                VoiceProviderEventRecord.provider_key == provider_key,
                VoiceProviderEventRecord.provider_event_id == provider_event_id,
            )
        )

    def get_or_create_call(
        self,
        *,
        provider_key: str,
        provider_call_id: str,
        line: VoiceLineConfigRecord,
        commit: bool = True,
    ) -> tuple[VoiceCallRecord, bool]:
        """Return (call, is_new) for an authenticated inbound call.

        The trusted line is authoritative for tenant ownership. Provider call
        identity is deduplicated by the database uniqueness constraint.
        No provider-event record is created by this method.
        """
        if (
            not isinstance(provider_key, str)
            or not provider_key.strip()
            or provider_key != provider_key.strip()
            or not isinstance(provider_call_id, str)
            or not provider_call_id.strip()
            or provider_call_id != provider_call_id.strip()
        ):
            raise ValueError("Provider call identity is invalid")

        if line.id is None or not isinstance(line.tenant_id, str) or not line.tenant_id.strip():
            raise VoiceLineNotEligibleError("Voice line has no persisted tenant identity")
        if line.line_status != "active":
            raise VoiceLineNotEligibleError("Voice line is not active")
        if not isinstance(line.provider_key, str) or line.provider_key != provider_key:
            raise VoiceLineNotEligibleError("Provider does not match configured line")
        if not isinstance(line.configured_phone_number, str) or not line.configured_phone_number:
            raise VoiceLineNotEligibleError("Voice line has no configured phone number")

        def validate_existing(call: VoiceCallRecord) -> None:
            if (
                call.tenant_id != line.tenant_id
                or call.voice_line_id != line.id
                or call.business_line_e164 != line.configured_phone_number
            ):
                raise VoiceLineNotEligibleError(
                    "Provider call is associated with another tenant or line"
                )

        try:
            existing = self._get_call(provider_key, provider_call_id)
            if existing is not None:
                validate_existing(existing)
                if commit:
                    self._session.commit()
                return existing, False

            now = datetime.now(UTC)
            with self._session.begin_nested():
                call = VoiceCallRecord(
                    call_id=str(uuid4()),
                    tenant_id=line.tenant_id,
                    voice_line_id=line.id,
                    provider_key=provider_key,
                    provider_call_id=provider_call_id,
                    business_line_e164=line.configured_phone_number,
                    status=VoiceCallStatus.RECEIVED.value,
                    started_at=now,
                    completed_at=None,
                    last_event_at=now,
                )
                self._session.add(call)
                self._session.flush()

            if commit:
                self._session.commit()
            return call, True

        except IntegrityError as exc:
            # Another request may have inserted this provider call first.
            existing = self._get_call(provider_key, provider_call_id)
            if existing is not None:
                try:
                    validate_existing(existing)
                except VoiceLineNotEligibleError:
                    if commit:
                        self._session.rollback()
                    raise

                if commit:
                    self._session.commit()
                return existing, False

            if commit:
                self._session.rollback()
            raise exc
        except Exception:
            if commit:
                self._session.rollback()
            raise

    def process_event(
        self,
        event: VoiceEvent,
        *,
        line: VoiceLineConfigRecord,
        commit: bool = True,
    ) -> tuple[VoiceCallRecord, VoiceProviderEventRecord, bool]:
        """Return (call, event record, is_new) after applying an event.

        ``line`` must come from the trusted phone-number registry.
        Event insertion and call-state changes are enclosed in a savepoint.
        """
        if line.id is None or not isinstance(line.tenant_id, str) or not line.tenant_id.strip():
            raise VoiceLineNotEligibleError("Voice line has no persisted tenant identity")
        if line.line_status != "active":
            raise VoiceLineNotEligibleError("Voice line is not active")
        if line.provider_key != event.provider_key:
            raise VoiceLineNotEligibleError("Event provider does not match configured line")
        if line.configured_phone_number != event.business_line_e164:
            raise VoiceLineNotEligibleError("Event called number does not match configured line")

        try:
            existing = self._get_event(event.provider_key, event.provider_event_id)
            if existing is not None:
                call = self._session.get(VoiceCallRecord, existing.call_id)
                if call is None:
                    raise RuntimeError("Persisted voice event references a missing call")
                if (
                    not self._matches_event(existing, event)
                    or existing.tenant_id != line.tenant_id
                    or call.tenant_id != line.tenant_id
                    or call.voice_line_id != line.id
                    or call.business_line_e164 != line.configured_phone_number
                ):
                    raise VoiceEventConflictError(
                        "Provider event ID conflicts with its persisted event or line"
                    )
                if commit:
                    self._session.commit()
                return call, existing, False

            with self._session.begin_nested():
                call = self._get_call(event.provider_key, event.provider_call_id)
                occurred_at = self._utc(event.occurred_at)

                if call is None:
                    status = next_call_status(None, event.event_type)
                    call = VoiceCallRecord(
                        call_id=str(uuid4()),
                        tenant_id=line.tenant_id,
                        voice_line_id=line.id,
                        provider_key=event.provider_key,
                        provider_call_id=event.provider_call_id,
                        business_line_e164=event.business_line_e164,
                        status=status.value,
                        started_at=occurred_at,
                        completed_at=(
                            occurred_at
                            if status
                            in {
                                VoiceCallStatus.COMPLETED,
                                VoiceCallStatus.FAILED,
                            }
                            else None
                        ),
                        last_event_at=occurred_at,
                    )
                    self._session.add(call)
                    self._session.flush()
                else:
                    if (
                        call.tenant_id != line.tenant_id
                        or call.voice_line_id != line.id
                        or call.business_line_e164 != line.configured_phone_number
                    ):
                        raise VoiceLineNotEligibleError(
                            "Provider call is associated with another line"
                        )

                    status = next_call_status(VoiceCallStatus(call.status), event.event_type)
                    call.status = status.value

                    if call.started_at is None or occurred_at < self._utc(call.started_at):
                        call.started_at = occurred_at

                    if (
                        status
                        in {
                            VoiceCallStatus.COMPLETED,
                            VoiceCallStatus.FAILED,
                        }
                        and call.completed_at is None
                    ):
                        call.completed_at = occurred_at

                    if occurred_at > self._utc(call.last_event_at):
                        call.last_event_at = occurred_at

                    call.updated_at = datetime.now(UTC)

                event_record = VoiceProviderEventRecord(
                    call_id=call.call_id,
                    tenant_id=line.tenant_id,
                    provider_key=event.provider_key,
                    provider_event_id=event.provider_event_id,
                    provider_call_id=event.provider_call_id,
                    event_type=event.event_type.value,
                    occurred_at=occurred_at,
                )
                self._session.add(event_record)
                self._session.flush()

            if commit:
                self._session.commit()
            return call, event_record, True

        except IntegrityError as exc:
            # A concurrent retry may have inserted this event first.
            duplicate = self._get_event(event.provider_key, event.provider_event_id)
            if duplicate is not None:
                call = self._session.get(VoiceCallRecord, duplicate.call_id)
                if (
                    not self._matches_event(duplicate, event)
                    or duplicate.tenant_id != line.tenant_id
                    or call is None
                    or call.tenant_id != line.tenant_id
                    or call.voice_line_id != line.id
                    or call.business_line_e164 != line.configured_phone_number
                ):
                    if commit:
                        self._session.rollback()
                    raise VoiceEventConflictError(
                        "Provider event ID conflicts with its persisted event or line"
                    ) from exc

                if commit:
                    self._session.commit()
                return call, duplicate, False

            if commit:
                self._session.rollback()
            raise
        except Exception:
            if commit:
                self._session.rollback()
            raise
