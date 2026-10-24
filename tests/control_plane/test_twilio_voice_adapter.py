from __future__ import annotations

from urllib.parse import urlencode

import pytest
from starlette.datastructures import FormData
from twilio.request_validator import RequestValidator

from app.control_plane.voice.adapters import (
    VoiceWebhookAuthenticationError,
    VoiceWebhookPayloadError,
    authenticate_and_normalize,
)
from app.control_plane.voice.models import VoiceEventType
from app.control_plane.voice.twilio_adapter import TwilioVoiceStatusAdapter


AUTH_TOKEN = "test-twilio-auth-token"
CALLBACK_URL = "https://voice.example.test/api/v1/voice/webhooks/twilio/status?source=prod"
TIMESTAMP = "2026-10-18T10:00:00Z"


def make_form_items(**overrides):
    values = {
        "CallSid": "CA123456789",
        "CallStatus": "in-progress",
        "To": "+14155550101",
        "From": "+14155550199",
        "SequenceNumber": "2",
        "Timestamp": TIMESTAMP,
    }
    values.update(overrides)
    return list(values.items())


def signed_request(items=None, *, signature_override=None):
    items = items if items is not None else make_form_items()
    body = urlencode(items).encode("utf-8")
    signature = RequestValidator(AUTH_TOKEN).compute_signature(
        CALLBACK_URL,
        FormData(items),
    )
    if signature_override is not None:
        signature = signature_override
    return {
        "request_url": CALLBACK_URL,
        "headers": {
            "Content-Type": "application/x-www-form-urlencoded",
            "X-Twilio-Signature": signature,
        },
        "body": body,
        "query_params": {"source": "prod"},
    }


@pytest.fixture
def adapter():
    return TwilioVoiceStatusAdapter(
        auth_token=AUTH_TOKEN,
        webhook_url=CALLBACK_URL,
    )


def test_valid_signature_normalizes_event(adapter):
    request = signed_request()

    event = authenticate_and_normalize(adapter, **request)

    assert event.provider_key == "twilio"
    assert event.provider_event_id == "CA123456789:2"
    assert event.provider_call_id == "CA123456789"
    assert event.business_line_e164 == "+14155550101"
    assert event.event_type == VoiceEventType.CALL_ANSWERED
    assert event.occurred_at.isoformat() == "2026-10-18T10:00:00+00:00"


def test_invalid_signature_is_rejected(adapter):
    with pytest.raises(VoiceWebhookAuthenticationError):
        authenticate_and_normalize(
            adapter,
            **signed_request(signature_override="invalid-signature"),
        )


def test_missing_signature_is_rejected(adapter):
    request = signed_request()
    request["headers"].pop("X-Twilio-Signature")

    with pytest.raises(VoiceWebhookAuthenticationError):
        authenticate_and_normalize(adapter, **request)


def test_callback_url_must_match_exact_configured_url(adapter):
    request = signed_request()
    request["request_url"] = "https://voice.example.test/api/v1/voice/webhooks/twilio/status"

    with pytest.raises(VoiceWebhookAuthenticationError):
        authenticate_and_normalize(adapter, **request)


def test_repeated_form_fields_are_signature_checked_and_rejected(adapter):
    items = make_form_items() + [("CallStatus", "completed")]
    request = signed_request(items)

    with pytest.raises(VoiceWebhookPayloadError, match="exactly once"):
        authenticate_and_normalize(adapter, **request)


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        ("initiated", VoiceEventType.CALL_RECEIVED),
        ("queued", VoiceEventType.CALL_RECEIVED),
        ("ringing", VoiceEventType.CALL_RECEIVED),
        ("in-progress", VoiceEventType.CALL_ANSWERED),
        ("completed", VoiceEventType.CALL_COMPLETED),
        ("busy", VoiceEventType.CALL_FAILED),
        ("failed", VoiceEventType.CALL_FAILED),
        ("no-answer", VoiceEventType.CALL_FAILED),
        ("canceled", VoiceEventType.CALL_FAILED),
    ],
)
def test_supported_statuses_map_to_canonical_event(adapter, status, expected):
    event = authenticate_and_normalize(
        adapter,
        **signed_request(make_form_items(CallStatus=status)),
    )

    assert event.event_type == expected


def test_unsupported_status_is_rejected_after_authentication(adapter):
    request = signed_request(make_form_items(CallStatus="unknown"))

    with pytest.raises(VoiceWebhookPayloadError, match="unsupported CallStatus"):
        authenticate_and_normalize(adapter, **request)


@pytest.mark.parametrize(
    "field",
    ["CallSid", "SequenceNumber", "Timestamp"],
)
def test_required_event_identity_and_timestamp_fields(adapter, field):
    items = [(key, value) for key, value in make_form_items() if key != field]

    with pytest.raises(VoiceWebhookPayloadError, match="requires CallSid"):
        authenticate_and_normalize(adapter, **signed_request(items))


def test_non_numeric_sequence_number_is_rejected(adapter):
    request = signed_request(make_form_items(SequenceNumber="sequence-two"))

    with pytest.raises(VoiceWebhookPayloadError, match="SequenceNumber"):
        authenticate_and_normalize(adapter, **request)


def test_timestamp_without_timezone_is_rejected(adapter):
    request = signed_request(make_form_items(Timestamp="2026-10-18T10:00:00"))

    with pytest.raises(VoiceWebhookPayloadError, match="timezone"):
        authenticate_and_normalize(adapter, **request)


def test_non_form_content_type_is_rejected(adapter):
    request = signed_request()
    request["headers"]["Content-Type"] = "application/json"

    with pytest.raises(VoiceWebhookPayloadError, match="form-encoded"):
        authenticate_and_normalize(adapter, **request)


def test_malformed_form_encoding_is_rejected(adapter):
    request = signed_request()
    request["body"] = b"CallSid=%FF&CallStatus=in-progress"

    with pytest.raises(VoiceWebhookPayloadError, match="malformed"):
        authenticate_and_normalize(adapter, **request)


def test_constructor_rejects_non_https_callback_url():
    with pytest.raises(ValueError, match="HTTPS"):
        TwilioVoiceStatusAdapter(
            auth_token=AUTH_TOKEN,
            webhook_url="http://voice.example.test/status",
        )


def test_event_id_is_stable_for_same_call_and_sequence(adapter):
    first = authenticate_and_normalize(
        adapter,
        **signed_request(make_form_items()),
    )
    second = authenticate_and_normalize(
        adapter,
        **signed_request(make_form_items()),
    )

    assert first.provider_event_id == second.provider_event_id
    assert first == second
