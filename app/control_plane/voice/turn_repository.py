from __future__ import annotations

import re
from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy import or_, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.control_plane.persistence.models import (
    VoiceCallRecord,
    VoiceCallTurnRecord,
)


class VoiceTurnNotFoundError(LookupError):
    """The requested turn does not exist within the specified tenant."""


class VoiceTurnConflictError(ValueError):
    """A turn callback conflicts with persisted identity or state."""


class VoiceTurnRepository:
    """Persist turn state transitions without storing recognized transcripts."""

    _FINGERPRINT_RE = re.compile(r"^[0-9a-f]{64}$")
    _MAX_RESPONSE_LENGTH = 8000

    def __init__(self, session: Session) -> None:
        self._session = session

    @staticmethod
    def _now() -> datetime:
        return datetime.now(UTC)

    @classmethod
    def _validate_fingerprint(cls, fingerprint: str) -> None:
        if not isinstance(fingerprint, str) or not cls._FINGERPRINT_RE.fullmatch(fingerprint):
            raise ValueError("request_fingerprint must be a lowercase SHA-256 hex digest")

    def get(
        self,
        *,
        tenant_id: str,
        turn_id: str,
        commit: bool = True,
    ) -> VoiceCallTurnRecord:
        record = self._session.scalar(
            select(VoiceCallTurnRecord).where(
                VoiceCallTurnRecord.turn_id == turn_id,
                VoiceCallTurnRecord.tenant_id == tenant_id,
            )
        )
        if record is None:
            raise VoiceTurnNotFoundError("Voice turn was not found")
        if commit:
            self._session.commit()
        return record

    def get_by_callback_token(
        self,
        callback_token: str,
        *,
        commit: bool = True,
    ) -> VoiceCallTurnRecord | None:
        record = self._session.scalar(
            select(VoiceCallTurnRecord).where(VoiceCallTurnRecord.callback_token == callback_token)
        )
        if commit:
            self._session.commit()
        return record

    def get_for_call_turn(
        self,
        *,
        tenant_id: str,
        call_id: str,
        turn_number: int,
        commit: bool = True,
    ) -> VoiceCallTurnRecord | None:
        """Find a turn using its tenant-scoped call identity."""
        if not tenant_id.strip():
            raise ValueError("tenant_id must not be blank")
        if not call_id.strip():
            raise ValueError("call_id must not be blank")
        if turn_number < 1:
            raise ValueError("turn_number must be positive")

        record = self._session.scalar(
            select(VoiceCallTurnRecord).where(
                VoiceCallTurnRecord.tenant_id == tenant_id,
                VoiceCallTurnRecord.call_id == call_id,
                VoiceCallTurnRecord.turn_number == turn_number,
            )
        )
        if commit:
            self._session.commit()
        return record

    def create_or_get(
        self,
        *,
        tenant_id: str,
        call_id: str,
        turn_number: int,
        callback_token: str,
        commit: bool = True,
    ) -> tuple[VoiceCallTurnRecord, bool]:
        """Return (turn, created); conflicting identities are rejected."""
        if not tenant_id.strip():
            raise ValueError("tenant_id must not be blank")
        if not call_id.strip():
            raise ValueError("call_id must not be blank")
        if turn_number < 1:
            raise ValueError("turn_number must be positive")
        if not callback_token or len(callback_token) > 128:
            raise ValueError("callback_token must contain 1 to 128 characters")

        try:
            call = self._session.scalar(
                select(VoiceCallRecord).where(
                    VoiceCallRecord.call_id == call_id,
                    VoiceCallRecord.tenant_id == tenant_id,
                )
            )
            if call is None:
                raise VoiceTurnNotFoundError("Voice call was not found within the specified tenant")

            existing = self._session.scalar(
                select(VoiceCallTurnRecord).where(
                    VoiceCallTurnRecord.tenant_id == tenant_id,
                    VoiceCallTurnRecord.call_id == call_id,
                    VoiceCallTurnRecord.turn_number == turn_number,
                )
            )
            if existing is not None:
                if existing.callback_token != callback_token:
                    raise VoiceTurnConflictError(
                        "Turn number is already associated with another callback token"
                    )
                if commit:
                    self._session.commit()
                return existing, False

            token_owner = self._session.scalar(
                select(VoiceCallTurnRecord).where(
                    VoiceCallTurnRecord.callback_token == callback_token
                )
            )
            if token_owner is not None:
                raise VoiceTurnConflictError(
                    "Callback token is already associated with another turn"
                )

            record = VoiceCallTurnRecord(
                turn_id=str(uuid4()),
                tenant_id=tenant_id,
                call_id=call_id,
                turn_number=turn_number,
                callback_token=callback_token,
                status="awaiting_input",
            )

            # A savepoint lets us handle uniqueness races without rolling back
            # unrelated work in the caller's outer transaction.
            with self._session.begin_nested():
                self._session.add(record)
                self._session.flush()

            if commit:
                self._session.commit()
            return record, True

        except IntegrityError as exc:
            # Another callback may have inserted the same turn or token first.
            existing = self._session.scalar(
                select(VoiceCallTurnRecord).where(
                    VoiceCallTurnRecord.tenant_id == tenant_id,
                    VoiceCallTurnRecord.call_id == call_id,
                    VoiceCallTurnRecord.turn_number == turn_number,
                )
            )
            if existing is not None and existing.callback_token == callback_token:
                if commit:
                    self._session.commit()
                return existing, False

            if commit:
                self._session.rollback()
            raise VoiceTurnConflictError(
                "Turn identity or callback token conflicts with persisted state"
            ) from exc
        except Exception:
            if commit:
                self._session.rollback()
            raise

    def mark_no_input(
        self,
        *,
        tenant_id: str,
        turn_id: str,
        commit: bool = True,
    ) -> tuple[VoiceCallTurnRecord, bool]:
        """Mark an unanswered turn as no_input; return (turn, changed)."""
        try:
            result = self._session.execute(
                update(VoiceCallTurnRecord)
                .where(
                    VoiceCallTurnRecord.turn_id == turn_id,
                    VoiceCallTurnRecord.tenant_id == tenant_id,
                    VoiceCallTurnRecord.status == "awaiting_input",
                )
                .values(
                    status="no_input",
                    completed_at=self._now(),
                    updated_at=self._now(),
                )
            )

            record = self._session.scalar(
                select(VoiceCallTurnRecord).where(
                    VoiceCallTurnRecord.turn_id == turn_id,
                    VoiceCallTurnRecord.tenant_id == tenant_id,
                )
            )
            if record is None:
                raise VoiceTurnNotFoundError("Voice turn was not found")

            changed = result.rowcount == 1
            if not changed and record.status != "no_input":
                raise VoiceTurnConflictError(
                    f"Cannot mark a turn with state '{record.status}' as no_input"
                )

            if commit:
                self._session.commit()
            return record, changed
        except Exception:
            if commit:
                self._session.rollback()
            raise

    def claim(
        self,
        *,
        tenant_id: str,
        turn_id: str,
        request_fingerprint: str,
        commit: bool = True,
    ) -> tuple[VoiceCallTurnRecord, bool]:
        """Atomically transition awaiting_input to processing.

        Returns (turn, claimed). Only one caller can successfully claim it.
        A duplicate fingerprint may inspect the existing state, but cannot
        claim a processing or completed turn again.
        """
        self._validate_fingerprint(request_fingerprint)

        try:
            result = self._session.execute(
                update(VoiceCallTurnRecord)
                .where(
                    VoiceCallTurnRecord.turn_id == turn_id,
                    VoiceCallTurnRecord.tenant_id == tenant_id,
                    VoiceCallTurnRecord.status == "awaiting_input",
                    or_(
                        VoiceCallTurnRecord.request_fingerprint.is_(None),
                        VoiceCallTurnRecord.request_fingerprint == request_fingerprint,
                    ),
                )
                .values(
                    status="processing",
                    request_fingerprint=request_fingerprint,
                    started_at=self._now(),
                    updated_at=self._now(),
                )
            )

            record = self._session.scalar(
                select(VoiceCallTurnRecord).where(
                    VoiceCallTurnRecord.turn_id == turn_id,
                    VoiceCallTurnRecord.tenant_id == tenant_id,
                )
            )
            if record is None:
                raise VoiceTurnNotFoundError("Voice turn was not found")

            if record.request_fingerprint not in (None, request_fingerprint):
                raise VoiceTurnConflictError(
                    "Turn was already submitted with different recognized input"
                )

            claimed = result.rowcount == 1
            if commit:
                self._session.commit()
            return record, claimed

        except Exception:
            if commit:
                self._session.rollback()
            raise

    def complete(
        self,
        *,
        tenant_id: str,
        turn_id: str,
        request_fingerprint: str,
        response_text: str,
        agent_run_id: str | None,
        commit: bool = True,
    ) -> tuple[VoiceCallTurnRecord, bool]:
        """Persist the bounded response; return (turn, newly_completed)."""
        self._validate_fingerprint(request_fingerprint)

        if not isinstance(response_text, str):
            raise ValueError("response_text must be a string")
        if len(response_text) > self._MAX_RESPONSE_LENGTH:
            raise ValueError("response_text exceeds 8000 characters")
        if agent_run_id is not None and len(agent_run_id) > 36:
            raise ValueError("agent_run_id exceeds 36 characters")

        try:
            result = self._session.execute(
                update(VoiceCallTurnRecord)
                .where(
                    VoiceCallTurnRecord.turn_id == turn_id,
                    VoiceCallTurnRecord.tenant_id == tenant_id,
                    VoiceCallTurnRecord.status == "processing",
                    VoiceCallTurnRecord.request_fingerprint == request_fingerprint,
                )
                .values(
                    status="completed",
                    response_text=response_text,
                    agent_run_id=agent_run_id,
                    failure_code=None,
                    completed_at=self._now(),
                    updated_at=self._now(),
                )
            )

            record = self._session.scalar(
                select(VoiceCallTurnRecord).where(
                    VoiceCallTurnRecord.turn_id == turn_id,
                    VoiceCallTurnRecord.tenant_id == tenant_id,
                )
            )
            if record is None:
                raise VoiceTurnNotFoundError("Voice turn was not found")

            if result.rowcount == 1:
                if commit:
                    self._session.commit()
                return record, True

            if record.request_fingerprint != request_fingerprint:
                raise VoiceTurnConflictError(
                    "Turn fingerprint does not match the completion request"
                )

            # Idempotent completion replay is allowed only for identical output.
            if (
                record.status == "completed"
                and record.response_text == response_text
                and record.agent_run_id == agent_run_id
            ):
                if commit:
                    self._session.commit()
                return record, False

            raise VoiceTurnConflictError(
                f"Cannot complete a turn currently in state '{record.status}'"
            )

        except Exception:
            if commit:
                self._session.rollback()
            raise

    def fail(
        self,
        *,
        tenant_id: str,
        turn_id: str,
        request_fingerprint: str,
        failure_code: str,
        commit: bool = True,
    ) -> tuple[VoiceCallTurnRecord, bool]:
        """Mark a processing turn failed without persisting exception details."""
        self._validate_fingerprint(request_fingerprint)
        if not failure_code or len(failure_code) > 100:
            raise ValueError("failure_code must contain 1 to 100 characters")

        try:
            result = self._session.execute(
                update(VoiceCallTurnRecord)
                .where(
                    VoiceCallTurnRecord.turn_id == turn_id,
                    VoiceCallTurnRecord.tenant_id == tenant_id,
                    VoiceCallTurnRecord.status == "processing",
                    VoiceCallTurnRecord.request_fingerprint == request_fingerprint,
                )
                .values(
                    status="failed",
                    failure_code=failure_code,
                    completed_at=self._now(),
                    updated_at=self._now(),
                )
            )

            record = self._session.scalar(
                select(VoiceCallTurnRecord).where(
                    VoiceCallTurnRecord.turn_id == turn_id,
                    VoiceCallTurnRecord.tenant_id == tenant_id,
                )
            )
            if record is None:
                raise VoiceTurnNotFoundError("Voice turn was not found")

            if result.rowcount == 1:
                if commit:
                    self._session.commit()
                return record, True

            if (
                record.status == "failed"
                and record.request_fingerprint == request_fingerprint
                and record.failure_code == failure_code
            ):
                if commit:
                    self._session.commit()
                return record, False

            if record.request_fingerprint != request_fingerprint:
                raise VoiceTurnConflictError("Turn fingerprint does not match the failure request")
            raise VoiceTurnConflictError(f"Cannot fail a turn currently in state '{record.status}'")

        except Exception:
            if commit:
                self._session.rollback()
            raise

    def claim_transient(
        self,
        *,
        tenant_id: str,
        turn_id: str,
        commit: bool = True,
    ) -> tuple[VoiceCallTurnRecord, bool, str | None]:
        """Claim a turn without persisting any transcript-derived value.

        Returns (turn, claimed, claim_token). The token is returned only
        to the caller that atomically claims the awaiting turn.
        """
        claim_token = str(uuid4())

        try:
            result = self._session.execute(
                update(VoiceCallTurnRecord)
                .where(
                    VoiceCallTurnRecord.turn_id == turn_id,
                    VoiceCallTurnRecord.tenant_id == tenant_id,
                    VoiceCallTurnRecord.status == "awaiting_input",
                )
                .values(
                    status="processing",
                    claim_token=claim_token,
                    request_fingerprint=None,
                    started_at=self._now(),
                    updated_at=self._now(),
                )
            )

            record = self._session.scalar(
                select(VoiceCallTurnRecord).where(
                    VoiceCallTurnRecord.turn_id == turn_id,
                    VoiceCallTurnRecord.tenant_id == tenant_id,
                )
            )
            if record is None:
                raise VoiceTurnNotFoundError("Voice turn was not found")

            claimed = result.rowcount == 1
            if commit:
                self._session.commit()

            return record, claimed, claim_token if claimed else None

        except Exception:
            if commit:
                self._session.rollback()
            raise

    def complete_transient(
        self,
        *,
        tenant_id: str,
        turn_id: str,
        claim_token: str,
        response_text: str,
        agent_run_id: str | None,
        commit: bool = True,
    ) -> tuple[VoiceCallTurnRecord, bool]:
        """Complete a transient-input turn only while holding its claim token.

        Transcript-derived input is not persisted. A stale or reused claim
        cannot complete the turn again.
        """
        if not isinstance(claim_token, str) or not claim_token.strip() or len(claim_token) > 36:
            raise ValueError("claim_token must contain 1 to 36 characters")

        if not isinstance(response_text, str):
            raise ValueError("response_text must be a string")

        if len(response_text) > self._MAX_RESPONSE_LENGTH:
            raise ValueError("response_text exceeds 8000 characters")

        if agent_run_id is not None and (
            not isinstance(agent_run_id, str) or not agent_run_id.strip() or len(agent_run_id) > 36
        ):
            raise ValueError("agent_run_id must contain 1 to 36 characters when provided")

        try:
            result = self._session.execute(
                update(VoiceCallTurnRecord)
                .where(
                    VoiceCallTurnRecord.turn_id == turn_id,
                    VoiceCallTurnRecord.tenant_id == tenant_id,
                    VoiceCallTurnRecord.status == "processing",
                    VoiceCallTurnRecord.claim_token == claim_token,
                    VoiceCallTurnRecord.request_fingerprint.is_(None),
                )
                .values(
                    status="completed",
                    response_text=response_text,
                    agent_run_id=agent_run_id,
                    failure_code=None,
                    completed_at=self._now(),
                    updated_at=self._now(),
                )
            )

            record = self._session.scalar(
                select(VoiceCallTurnRecord).where(
                    VoiceCallTurnRecord.turn_id == turn_id,
                    VoiceCallTurnRecord.tenant_id == tenant_id,
                )
            )

            if record is None:
                raise VoiceTurnNotFoundError("Voice turn was not found")

            if result.rowcount != 1:
                raise VoiceTurnConflictError(
                    "Turn is no longer processing under this transient claim"
                )

            if commit:
                self._session.commit()

            return record, True

        except Exception:
            if commit:
                self._session.rollback()
            raise

    def fail_transient(
        self,
        *,
        tenant_id: str,
        turn_id: str,
        claim_token: str,
        failure_code: str,
        commit: bool = True,
    ) -> tuple[VoiceCallTurnRecord, bool]:
        """Fail a transient-input turn only while holding its claim token."""
        if not isinstance(claim_token, str) or not claim_token.strip() or len(claim_token) > 36:
            raise ValueError("claim_token must contain 1 to 36 characters")

        if not isinstance(failure_code, str) or not failure_code.strip() or len(failure_code) > 100:
            raise ValueError("failure_code must contain 1 to 100 characters")

        try:
            result = self._session.execute(
                update(VoiceCallTurnRecord)
                .where(
                    VoiceCallTurnRecord.turn_id == turn_id,
                    VoiceCallTurnRecord.tenant_id == tenant_id,
                    VoiceCallTurnRecord.status == "processing",
                    VoiceCallTurnRecord.claim_token == claim_token,
                )
                .values(
                    status="failed",
                    failure_code=failure_code,
                    completed_at=self._now(),
                    updated_at=self._now(),
                )
            )

            record = self._session.scalar(
                select(VoiceCallTurnRecord).where(
                    VoiceCallTurnRecord.turn_id == turn_id,
                    VoiceCallTurnRecord.tenant_id == tenant_id,
                )
            )
            if record is None:
                raise VoiceTurnNotFoundError("Voice turn was not found")
            if result.rowcount != 1:
                raise VoiceTurnConflictError("Turn is no longer owned by this claim")

            if commit:
                self._session.commit()
            return record, True

        except Exception:
            if commit:
                self._session.rollback()
            raise
