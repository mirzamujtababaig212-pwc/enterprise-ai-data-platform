from datetime import UTC, datetime

import pytest

from app.control_plane.voice.adapters import (
    VoiceWebhookAuthenticationError,
    VoiceWebhookPayloadError,
    authenticate_and_normalize,
)
from app.control_plane.voice.models import VoiceEvent, VoiceEventType


class FakeVoiceProviderAdapter:
    provider_key = "test-provider"

    def __init__(self, event=None, error=None):
        self.event = event
        self.error = error
        self.received_request = None

    def authenticate_and_parse(
        self,
        *,
        request_url,
        headers,
        body,
        query_params,
    ):
        self.received_request = {
            "request_url": request_url,
            "headers": headers,
            "body": body,
            "query_params": query_params,
        }
        if self.error is not None:
            raise self.error
        return self.event


def canonical_event(
    *,
    provider_key="test-provider",
) -> VoiceEvent:
    return VoiceEvent(
        provider_key=provider_key,
        provider_event_id="event-001",
        provider_call_id="call-001",
        business_line_e164="+14155552671",
        event_type=VoiceEventType.CALL_RECEIVED,
        occurred_at=datetime.now(UTC),
    )


def test_adapter_contract_preserves_raw_request_and_returns_event():
    adapter = FakeVoiceProviderAdapter(event=canonical_event())
    headers = {"x-test-signature": "test-signature"}
    body = b'{"event":"call.received"}'
    query_params = {"source": "test"}

    result = authenticate_and_normalize(
        adapter,
        request_url="https://voice.example.test/webhook",
        headers=headers,
        body=body,
        query_params=query_params,
    )

    assert result == adapter.event
    assert adapter.received_request == {
        "request_url": "https://voice.example.test/webhook",
        "headers": headers,
        "body": body,
        "query_params": query_params,
    }


def test_adapter_rejects_event_from_different_provider():
    adapter = FakeVoiceProviderAdapter(event=canonical_event(provider_key="another-provider"))

    with pytest.raises(
        VoiceWebhookPayloadError,
        match="does not match",
    ):
        authenticate_and_normalize(
            adapter,
            request_url="https://voice.example.test/webhook",
            headers={},
            body=b"{}",
            query_params={},
        )


def test_adapter_propagates_signature_authentication_failure():
    adapter = FakeVoiceProviderAdapter(error=VoiceWebhookAuthenticationError("Invalid signature."))

    with pytest.raises(VoiceWebhookAuthenticationError):
        authenticate_and_normalize(
            adapter,
            request_url="https://voice.example.test/webhook",
            headers={},
            body=b"{}",
            query_params={},
        )


def test_adapter_propagates_invalid_provider_payload():
    adapter = FakeVoiceProviderAdapter(error=VoiceWebhookPayloadError("Unsupported event."))

    with pytest.raises(VoiceWebhookPayloadError, match="Unsupported event"):
        authenticate_and_normalize(
            adapter,
            request_url="https://voice.example.test/webhook",
            headers={},
            body=b"{}",
            query_params={},
        )


def test_adapter_rejects_non_bytes_body_before_invocation():
    adapter = FakeVoiceProviderAdapter(event=canonical_event())

    with pytest.raises(VoiceWebhookPayloadError, match="raw bytes"):
        authenticate_and_normalize(
            adapter,
            request_url="https://voice.example.test/webhook",
            headers={},
            body='{"event":"call.received"}',
            query_params={},
        )

    assert adapter.received_request is None


def test_adapter_rejects_noncanonical_return_value():
    adapter = FakeVoiceProviderAdapter(event={"provider_key": "test-provider"})

    with pytest.raises(
        VoiceWebhookPayloadError,
        match="canonical voice event",
    ):
        authenticate_and_normalize(
            adapter,
            request_url="https://voice.example.test/webhook",
            headers={},
            body=b"{}",
            query_params={},
        )
