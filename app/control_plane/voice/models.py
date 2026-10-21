from __future__ import annotations

import re
from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator


_E164_PATTERN = re.compile(r"^\+[1-9][0-9]{1,14}$")


class VoiceCallStatus(StrEnum):
    RECEIVED = "received"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class VoiceEventType(StrEnum):
    CALL_RECEIVED = "call_received"
    CALL_ANSWERED = "call_answered"
    CALL_COMPLETED = "call_completed"
    CALL_FAILED = "call_failed"


class VoiceEvent(BaseModel):
    """Canonical event emitted by an internal provider adapter.

    Provider-specific webhook payloads must be authenticated and normalized
    before they are represented by this model.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    provider_key: str = Field(min_length=1, max_length=100)
    provider_event_id: str = Field(min_length=1, max_length=255)
    provider_call_id: str = Field(min_length=1, max_length=255)
    business_line_e164: str
    event_type: VoiceEventType
    occurred_at: datetime

    @field_validator(
        "provider_key",
        "provider_event_id",
        "provider_call_id",
        mode="before",
    )
    @classmethod
    def validate_identifier(cls, value: object) -> object:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Identifier must be a non-empty string")
        if value != value.strip():
            raise ValueError("Identifier must not contain surrounding whitespace")
        return value

    @field_validator("business_line_e164")
    @classmethod
    def validate_business_line(cls, value: str) -> str:
        if not _E164_PATTERN.fullmatch(value):
            raise ValueError("Business line must use E.164 format")
        return value


class VoiceCall(BaseModel):
    """Minimal call state; deliberately excludes transcripts and raw payloads."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    call_id: str = Field(min_length=1, max_length=36)
    tenant_id: str = Field(min_length=1, max_length=255)
    provider_key: str = Field(min_length=1, max_length=100)
    provider_call_id: str = Field(min_length=1, max_length=255)
    business_line_e164: str
    status: VoiceCallStatus
    started_at: datetime | None = None
    completed_at: datetime | None = None
    last_event_at: datetime
