from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

_PHONE_PATTERN = r"^\+[1-9][0-9]{1,14}$"


class VoiceLineCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    configured_phone_number: str = Field(min_length=3, max_length=16, pattern=_PHONE_PATTERN)
    carrier_trunk_line: str = Field(min_length=1, max_length=255)
    call_forwarding_code: str | None = Field(default=None, max_length=128)

    inbound_opening_greeting: str = Field(min_length=1, max_length=10000)
    inbound_system_prompt_override: str | None = Field(default=None, max_length=20000)
    outbound_opening_message: str = Field(min_length=1, max_length=10000)
    outbound_system_prompt_override: str | None = Field(default=None, max_length=20000)

    voice_synthesizer: str = Field(default="polly-kajal", min_length=1, max_length=255)
    spoken_language: str = Field(default="hi-IN", min_length=2, max_length=20)
    conversational_pause_pacing: float = Field(default=1.0, ge=0, le=10)
    barge_in_sensitivity: int = Field(default=2, ge=1, le=10)
    llm_reasoning_model: str = Field(default="openai/gpt-4o-mini", min_length=1, max_length=255)
    max_call_duration_minutes: int = Field(default=10, ge=1, le=240)


class VoiceLineUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    configured_phone_number: str | None = Field(
        default=None, min_length=3, max_length=16, pattern=_PHONE_PATTERN
    )
    carrier_trunk_line: str | None = Field(default=None, min_length=1, max_length=255)
    call_forwarding_code: str | None = Field(default=None, max_length=128)

    inbound_opening_greeting: str | None = Field(default=None, min_length=1, max_length=10000)
    inbound_system_prompt_override: str | None = Field(default=None, max_length=20000)
    outbound_opening_message: str | None = Field(default=None, min_length=1, max_length=10000)
    outbound_system_prompt_override: str | None = Field(default=None, max_length=20000)

    voice_synthesizer: str | None = Field(default=None, min_length=1, max_length=255)
    spoken_language: str | None = Field(default=None, min_length=2, max_length=20)
    conversational_pause_pacing: float | None = Field(default=None, ge=0, le=10)
    barge_in_sensitivity: int | None = Field(default=None, ge=1, le=10)
    llm_reasoning_model: str | None = Field(default=None, min_length=1, max_length=255)
    max_call_duration_minutes: int | None = Field(default=None, ge=1, le=240)


class VoiceLineResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True, extra="forbid")

    id: int
    tenant_id: str
    configured_phone_number: str
    carrier_trunk_line: str
    call_forwarding_code: str | None

    line_status: str
    inbound_opening_greeting: str
    inbound_system_prompt_override: str | None
    outbound_opening_message: str
    outbound_system_prompt_override: str | None

    provider_key: str | None
    agent_uuid: str | None
    provider_sync_status: str
    last_provider_sync_at: datetime | None
    last_provider_sync_error_code: str | None

    voice_synthesizer: str
    spoken_language: str
    conversational_pause_pacing: float
    barge_in_sensitivity: int
    llm_reasoning_model: str
    max_call_duration_minutes: int

    created_at: datetime
    updated_at: datetime
