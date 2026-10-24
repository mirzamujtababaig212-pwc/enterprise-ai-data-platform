from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from urllib.parse import parse_qsl, urlsplit

from starlette.datastructures import FormData
from twilio.request_validator import RequestValidator

from app.control_plane.voice.adapters import (
    VoiceWebhookAuthenticationError,
    VoiceWebhookPayloadError,
)
from app.control_plane.voice.models import VoiceEvent, VoiceEventType


class TwilioVoiceStatusAdapter:
    """Authenticate and normalize Twilio form-encoded voice status callbacks.

    ``webhook_url`` must be the exact externally configured HTTPS callback URL.
    It is intentionally not reconstructed from request or forwarded headers.
    """

    provider_key = "twilio"
    MAX_BODY_BYTES = 64 * 1024

    _EVENT_TYPES = {
        "initiated": VoiceEventType.CALL_RECEIVED,
        "queued": VoiceEventType.CALL_RECEIVED,
        "ringing": VoiceEventType.CALL_RECEIVED,
        "in-progress": VoiceEventType.CALL_ANSWERED,
        "completed": VoiceEventType.CALL_COMPLETED,
        "busy": VoiceEventType.CALL_FAILED,
        "failed": VoiceEventType.CALL_FAILED,
        "no-answer": VoiceEventType.CALL_FAILED,
        "canceled": VoiceEventType.CALL_FAILED,
    }

    def __init__(self, *, auth_token: str, webhook_url: str) -> None:
        if not isinstance(auth_token, str) or not auth_token.strip():
            raise ValueError("Twilio auth token is not configured")
        if not isinstance(webhook_url, str):
            raise ValueError("Twilio webhook URL is not configured")

        parsed = urlsplit(webhook_url)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.fragment
        ):
            raise ValueError("Twilio webhook URL must be an absolute HTTPS URL")

        self._webhook_url = webhook_url
        self._validator = RequestValidator(auth_token)

    @staticmethod
    def _header(headers: Mapping[str, str], name: str) -> str | None:
        for key, value in headers.items():
            if key.lower() == name.lower():
                return value
        return None

    @staticmethod
    def _parse_timestamp(value: str) -> datetime:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            try:
                parsed = parsedate_to_datetime(value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise VoiceWebhookPayloadError("Twilio callback timestamp is invalid.") from exc

        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise VoiceWebhookPayloadError("Twilio callback timestamp must include a timezone.")
        return parsed.astimezone(UTC)

    def authenticate_and_parse(
        self,
        *,
        request_url: str,
        headers: Mapping[str, str],
        body: bytes,
        query_params: Mapping[str, str],
    ) -> VoiceEvent:
        if request_url != self._webhook_url:
            raise VoiceWebhookAuthenticationError(
                "Webhook URL does not match the configured callback URL."
            )

        content_type = self._header(headers, "content-type") or ""
        if content_type.split(";", 1)[0].strip().lower() != ("application/x-www-form-urlencoded"):
            raise VoiceWebhookPayloadError("Twilio status callbacks must use form-encoded content.")

        if not isinstance(body, bytes) or len(body) > self.MAX_BODY_BYTES:
            raise VoiceWebhookPayloadError("Twilio callback body is invalid or too large.")

        try:
            form_items = parse_qsl(
                body.decode("utf-8"),
                keep_blank_values=True,
                strict_parsing=True,
                encoding="utf-8",
                errors="strict",
            )
        except (UnicodeDecodeError, ValueError) as exc:
            raise VoiceWebhookPayloadError("Twilio callback form body is malformed.") from exc

        if not form_items:
            raise VoiceWebhookPayloadError("Twilio callback form body is empty.")

        # FormData preserves repeated keys. Twilio's validator calls getall()
        # where available, so repeated form fields remain part of the signature.
        form = FormData(form_items)
        signature = self._header(headers, "x-twilio-signature")
        if not signature or not self._validator.validate(
            self._webhook_url,
            form,
            signature,
        ):
            raise VoiceWebhookAuthenticationError("Twilio webhook signature is missing or invalid.")

        # Do not normalize any event fields until the signature is verified.
        values: dict[str, str] = {}
        for key in form.keys():
            all_values = form.getlist(key)
            if len(all_values) != 1:
                raise VoiceWebhookPayloadError(
                    f"Twilio callback field {key!r} must occur exactly once."
                )
            values[key] = all_values[0]

        call_sid = values.get("CallSid", "")
        status = values.get("CallStatus", "").lower()
        called_number = values.get("To", "")
        sequence_number = values.get("SequenceNumber", "")
        timestamp = values.get("Timestamp", "")

        if not call_sid or not sequence_number or not timestamp:
            raise VoiceWebhookPayloadError(
                "Twilio callback requires CallSid, SequenceNumber, and Timestamp."
            )
        if not sequence_number.isascii() or not sequence_number.isdigit():
            raise VoiceWebhookPayloadError(
                "Twilio callback SequenceNumber must be a non-negative integer."
            )
        if status not in self._EVENT_TYPES:
            raise VoiceWebhookPayloadError("Twilio callback contains an unsupported CallStatus.")

        try:
            return VoiceEvent(
                provider_key=self.provider_key,
                provider_event_id=f"{call_sid}:{sequence_number}",
                provider_call_id=call_sid,
                business_line_e164=called_number,
                event_type=self._EVENT_TYPES[status],
                occurred_at=self._parse_timestamp(timestamp),
            )
        except VoiceWebhookPayloadError:
            raise
        except (TypeError, ValueError) as exc:
            raise VoiceWebhookPayloadError(
                "Twilio callback fields could not be normalized."
            ) from exc
