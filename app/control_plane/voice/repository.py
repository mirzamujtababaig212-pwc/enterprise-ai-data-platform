from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.control_plane.persistence.models import VoiceLineConfigRecord


class VoiceLineNotFoundError(LookupError):
    """A voice line does not exist for the requested tenant."""


class DuplicateVoiceLinePhoneNumberError(ValueError):
    """A configured phone number is already assigned to a voice line."""


class VoiceLineRepository:
    """Tenant-scoped persistence for voice-line configuration.

    The caller supplies the SQLAlchemy session. Set commit=False when a
    surrounding application service owns the transaction.
    """

    _MUTABLE_FIELDS = frozenset(
        {
            "configured_phone_number",
            "carrier_trunk_line",
            "call_forwarding_code",
            "line_status",
            "inbound_opening_greeting",
            "inbound_system_prompt_override",
            "outbound_opening_message",
            "outbound_system_prompt_override",
            "provider_key",
            "agent_uuid",
            "provider_sync_status",
            "last_provider_sync_at",
            "last_provider_sync_error_code",
            "voice_synthesizer",
            "spoken_language",
            "conversational_pause_pacing",
            "barge_in_sensitivity",
            "llm_reasoning_model",
            "max_call_duration_minutes",
        }
    )

    def __init__(self, session: Session) -> None:
        self._session = session

    @classmethod
    def _validate_values(
        cls,
        values: Mapping[str, Any],
        *,
        allow_empty: bool = False,
    ) -> dict[str, Any]:
        if not isinstance(values, Mapping):
            raise TypeError("Voice-line values must be a mapping")

        result = dict(values)
        forbidden = {"id", "tenant_id", "created_at", "updated_at"}
        supplied_forbidden = forbidden.intersection(result)
        if supplied_forbidden:
            raise ValueError(
                "Protected voice-line fields cannot be assigned: "
                + ", ".join(sorted(supplied_forbidden))
            )

        unknown = set(result).difference(cls._MUTABLE_FIELDS)
        if unknown:
            raise ValueError("Unsupported voice-line fields: " + ", ".join(sorted(unknown)))

        if not result and not allow_empty:
            raise ValueError("At least one voice-line field must be supplied")

        return result

    @staticmethod
    def _is_phone_number_conflict(exc: IntegrityError) -> bool:
        # PostgreSQL drivers may expose the violated constraint name.
        diag = getattr(exc.orig, "diag", None)
        constraint_name = getattr(diag, "constraint_name", None)
        if constraint_name == "uq_voice_line_configs_configured_phone_number":
            return True

        # SQLite's integrity error does not expose PostgreSQL constraint names.
        message = str(exc.orig).lower()
        return "voice_line_configs.configured_phone_number" in message and "unique" in message

    def _flush_and_finish(self, *, commit: bool) -> None:
        try:
            self._session.flush()
            if commit:
                self._session.commit()
        except IntegrityError as exc:
            if commit:
                self._session.rollback()
                if self._is_phone_number_conflict(exc):
                    raise DuplicateVoiceLinePhoneNumberError(
                        "Configured phone number is already assigned"
                    ) from exc
            raise
        except Exception:
            if commit:
                self._session.rollback()
            raise

    def create(
        self,
        *,
        tenant_id: str,
        values: Mapping[str, Any],
        commit: bool = True,
    ) -> VoiceLineConfigRecord:
        if not isinstance(tenant_id, str) or not tenant_id.strip():
            raise ValueError("tenant_id must be a non-empty string")

        clean_values = self._validate_values(values)
        record = VoiceLineConfigRecord(
            tenant_id=tenant_id,
            **clean_values,
        )
        self._session.add(record)
        self._flush_and_finish(commit=commit)
        return record

    def get_by_phone_number(
        self,
        phone_number: str,
    ) -> VoiceLineConfigRecord | None:
        """Resolve an inbound called number to its trusted line configuration.

        Phone numbers are globally unique in the voice-line registry. The
        returned record supplies the authoritative tenant and line status;
        callers must not derive either value from webhook payload fields.
        This lookup intentionally does not filter by line status so the
        caller can explicitly reject paused or offline lines.
        """
        if not isinstance(phone_number, str) or not phone_number.strip():
            raise ValueError("phone_number must be a non-empty string")

        return self._session.scalar(
            select(VoiceLineConfigRecord).where(
                VoiceLineConfigRecord.configured_phone_number == phone_number,
            )
        )

    def get_for_tenant(
        self,
        line_id: int,
        tenant_id: str,
    ) -> VoiceLineConfigRecord | None:
        return self._session.scalar(
            select(VoiceLineConfigRecord).where(
                VoiceLineConfigRecord.id == line_id,
                VoiceLineConfigRecord.tenant_id == tenant_id,
            )
        )

    def list_for_tenant(
        self,
        tenant_id: str,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> list[VoiceLineConfigRecord]:
        if not isinstance(tenant_id, str) or not tenant_id.strip():
            raise ValueError("tenant_id must be a non-empty string")
        if not 1 <= limit <= 500:
            raise ValueError("limit must be between 1 and 500")
        if offset < 0:
            raise ValueError("offset must be non-negative")

        statement = (
            select(VoiceLineConfigRecord)
            .where(VoiceLineConfigRecord.tenant_id == tenant_id)
            .order_by(
                VoiceLineConfigRecord.updated_at.desc(),
                VoiceLineConfigRecord.id.desc(),
            )
            .limit(limit)
            .offset(offset)
        )
        return list(self._session.scalars(statement).all())

    def update_for_tenant(
        self,
        line_id: int,
        *,
        tenant_id: str,
        changes: Mapping[str, Any],
        commit: bool = True,
    ) -> VoiceLineConfigRecord:
        clean_changes = self._validate_values(changes)
        record = self.get_for_tenant(line_id, tenant_id)

        if record is None:
            raise VoiceLineNotFoundError("Voice line not found")

        for field, value in clean_changes.items():
            setattr(record, field, value)

        # Set this explicitly so timestamp behavior is predictable in tests
        # and on databases where the ORM-side onupdate hook is not invoked.
        record.updated_at = datetime.now(UTC)

        self._flush_and_finish(commit=commit)
        return record
