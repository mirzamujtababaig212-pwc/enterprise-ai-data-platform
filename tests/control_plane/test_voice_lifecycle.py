from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from app.control_plane.voice.lifecycle import next_call_status
from app.control_plane.voice.models import (
    VoiceCallStatus,
    VoiceEvent,
    VoiceEventType,
)


@pytest.mark.parametrize(
    ("current", "event", "expected"),
    [
        (None, VoiceEventType.CALL_RECEIVED, VoiceCallStatus.RECEIVED),
        (VoiceCallStatus.RECEIVED, VoiceEventType.CALL_RECEIVED, VoiceCallStatus.RECEIVED),
        (VoiceCallStatus.RECEIVED, VoiceEventType.CALL_ANSWERED, VoiceCallStatus.IN_PROGRESS),
        (VoiceCallStatus.IN_PROGRESS, VoiceEventType.CALL_ANSWERED, VoiceCallStatus.IN_PROGRESS),
        (VoiceCallStatus.RECEIVED, VoiceEventType.CALL_COMPLETED, VoiceCallStatus.COMPLETED),
        (VoiceCallStatus.IN_PROGRESS, VoiceEventType.CALL_FAILED, VoiceCallStatus.FAILED),
    ],
)
def test_next_call_status_transitions(current, event, expected):
    assert next_call_status(current, event) is expected


@pytest.mark.parametrize(
    "terminal",
    [VoiceCallStatus.COMPLETED, VoiceCallStatus.FAILED],
)
@pytest.mark.parametrize("late_event", list(VoiceEventType))
def test_terminal_state_does_not_regress(terminal, late_event):
    assert next_call_status(terminal, late_event) is terminal


def test_event_accepts_canonical_provider_neutral_fields():
    event = VoiceEvent(
        provider_key="adapter-a",
        provider_event_id="event-001",
        provider_call_id="call-001",
        business_line_e164="+14155552671",
        event_type=VoiceEventType.CALL_RECEIVED,
        occurred_at=datetime.now(UTC),
    )

    assert event.business_line_e164 == "+14155552671"
    assert event.event_type is VoiceEventType.CALL_RECEIVED


@pytest.mark.parametrize(
    "phone",
    [
        "14155552671",
        "+0123456789",
        "+1 415 555 2671",
        "+1234567890123456",
    ],
)
def test_event_rejects_invalid_business_line(phone):
    with pytest.raises(ValidationError):
        VoiceEvent(
            provider_key="adapter-a",
            provider_event_id="event-001",
            provider_call_id="call-001",
            business_line_e164=phone,
            event_type=VoiceEventType.CALL_RECEIVED,
            occurred_at=datetime.now(UTC),
        )


def test_event_rejects_extra_provider_payload_fields():
    with pytest.raises(ValidationError):
        VoiceEvent(
            provider_key="adapter-a",
            provider_event_id="event-001",
            provider_call_id="call-001",
            business_line_e164="+14155552671",
            event_type=VoiceEventType.CALL_RECEIVED,
            occurred_at=datetime.now(UTC),
            raw_payload={"transcript": "must not be accepted"},
        )
