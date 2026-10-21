from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

from app.control_plane.voice.models import VoiceEvent


class VoiceWebhookAuthenticationError(ValueError):
    """The provider webhook could not be authenticated."""


class VoiceWebhookPayloadError(ValueError):
    """The authenticated provider payload could not be normalized safely."""


class VoiceProviderAdapter(Protocol):
    """Contract implemented by each telephony-provider adapter.

    Implementations must authenticate the original request before parsing
    its body into a canonical event. They must not trust tenant identity
    or agent configuration supplied by the provider payload.
    """

    provider_key: str

    def authenticate_and_parse(
        self,
        *,
        request_url: str,
        headers: Mapping[str, str],
        body: bytes,
        query_params: Mapping[str, str],
    ) -> VoiceEvent:
        """Authenticate a raw webhook request and return its canonical event."""
        ...


def authenticate_and_normalize(
    adapter: VoiceProviderAdapter,
    *,
    request_url: str,
    headers: Mapping[str, str],
    body: bytes,
    query_params: Mapping[str, str],
) -> VoiceEvent:
    """Apply the common adapter boundary and enforce provider identity.

    Provider-specific signature verification belongs inside
    ``authenticate_and_parse`` and must happen before payload parsing.
    """
    if not isinstance(body, bytes):
        raise VoiceWebhookPayloadError("Webhook body must be raw bytes.")

    if not isinstance(adapter.provider_key, str) or not adapter.provider_key.strip():
        raise VoiceWebhookPayloadError("Adapter provider identity is invalid.")

    if not isinstance(request_url, str) or not request_url.startswith("https://"):
        raise VoiceWebhookPayloadError("Webhook request URL must be an absolute HTTPS URL.")

    event = adapter.authenticate_and_parse(
        request_url=request_url,
        headers=headers,
        body=body,
        query_params=query_params,
    )

    if not isinstance(event, VoiceEvent):
        raise VoiceWebhookPayloadError("Provider adapter did not return a canonical voice event.")

    if event.provider_key != adapter.provider_key:
        raise VoiceWebhookPayloadError("Canonical event provider does not match the adapter.")

    return event
