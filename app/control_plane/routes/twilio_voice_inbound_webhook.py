from __future__ import annotations

import secrets
from typing import Annotated
from urllib.parse import parse_qsl, urlencode, urlsplit

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from sqlalchemy.orm import Session
from starlette.datastructures import FormData
from twilio.request_validator import RequestValidator
from twilio.twiml.voice_response import VoiceResponse

from ai_platform.llm_gateway.config.settings import settings
from app.control_plane.persistence.database import get_db
from app.control_plane.persistence.models import VoiceCallRecord
from app.control_plane.voice.call_repository import (
    VoiceCallRepository,
    VoiceLineNotEligibleError,
)
from app.control_plane.voice.repository import VoiceLineRepository
from app.control_plane.voice.turn_repository import (
    VoiceTurnConflictError,
    VoiceTurnRepository,
)

router = APIRouter(tags=["voice-webhooks"])

INBOUND_PATH = "/api/v1/voice/webhooks/twilio/inbound"
SPEECH_PATH = "/api/v1/voice/webhooks/twilio/speech"
MAX_BODY_BYTES = 64 * 1024


def _configured_url(raw_url: str, expected_path: str) -> tuple[str, str]:
    """Validate a configured public HTTPS callback URL."""
    if not raw_url or not raw_url.isascii():
        raise ValueError("Webhook URL is not configured correctly")

    parts = urlsplit(raw_url)
    if (
        parts.scheme != "https"
        or not parts.hostname
        or parts.username is not None
        or parts.password is not None
        or parts.fragment
        or parts.path != expected_path
    ):
        raise ValueError("Webhook URL is not configured correctly")

    # Accessing .port also validates malformed port syntax.
    try:
        _ = parts.port
    except ValueError as exc:
        raise ValueError("Webhook URL is not configured correctly") from exc

    return raw_url, parts.query


async def _read_bounded_body(request: Request) -> bytes:
    content_type = request.headers.get("content-type", "")
    if content_type.split(";", 1)[0].strip().lower() != ("application/x-www-form-urlencoded"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Expected a form-encoded voice webhook.",
        )

    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            if int(content_length) < 0:
                raise ValueError
            if int(content_length) > MAX_BODY_BYTES:
                raise HTTPException(
                    status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                    detail="Voice webhook body is too large.",
                )
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid voice webhook content length.",
            ) from exc

    chunks: list[bytes] = []
    size = 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > MAX_BODY_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                detail="Voice webhook body is too large.",
            )
        chunks.append(chunk)

    return b"".join(chunks)


def _parse_form(body: bytes) -> list[tuple[str, str]]:
    try:
        return parse_qsl(
            body.decode("utf-8"),
            keep_blank_values=True,
            strict_parsing=False,
            max_num_fields=1000,
            encoding="utf-8",
            errors="strict",
        )
    except (UnicodeDecodeError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid voice webhook form.",
        ) from exc


def _validate_twilio_request(
    *,
    configured_url: str,
    body: bytes,
    signature: str | None,
    auth_token: str,
) -> list[tuple[str, str]]:
    if not auth_token:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Voice webhook is not configured.",
        )

    if not signature:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid voice webhook signature.",
        )

    items = _parse_form(body)
    try:
        valid = RequestValidator(auth_token).validate(
            configured_url,
            FormData(items),
            signature,
        )
    except (TypeError, ValueError):
        valid = False

    if not valid:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid voice webhook signature.",
        )

    return items


def _one_field(items: list[tuple[str, str]], name: str) -> str:
    values = [value for key, value in items if key == name]
    if len(values) != 1 or not values[0].strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid voice webhook payload.",
        )
    return values[0]


def _speech_action_url(turn_token: str) -> str:
    base_url, configured_query = _configured_url(
        settings.TWILIO_SPEECH_WEBHOOK_URL,
        SPEECH_PATH,
    )
    # The token is an opaque lookup key, not a substitute for Twilio
    # signature verification. The speech endpoint must validate both.
    query_items = parse_qsl(configured_query, keep_blank_values=True)
    if any(key == "turn_token" for key, _ in query_items):
        raise ValueError("Speech URL must not preconfigure turn_token")
    query_items.append(("turn_token", turn_token))
    return f"{base_url.split('?', 1)[0]}?{urlencode(query_items)}"


@router.post(INBOUND_PATH, status_code=status.HTTP_200_OK)
async def twilio_inbound_call(
    request: Request,
    db: Annotated[Session, Depends(get_db)],
) -> Response:
    try:
        callback_url, configured_query = _configured_url(
            settings.TWILIO_INBOUND_WEBHOOK_URL,
            INBOUND_PATH,
        )
        speech_url, _ = _configured_url(
            settings.TWILIO_SPEECH_WEBHOOK_URL,
            SPEECH_PATH,
        )
        del speech_url
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Voice webhook is not configured.",
        ) from exc

    # Never build the signed URL from Host or forwarded headers.
    raw_path = request.scope.get("raw_path", b"")
    raw_query = request.scope.get("query_string", b"")
    if raw_path != INBOUND_PATH.encode("ascii") or raw_query != configured_query.encode("ascii"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid voice webhook request.",
        )

    body = await _read_bounded_body(request)
    items = _validate_twilio_request(
        configured_url=callback_url,
        body=body,
        signature=request.headers.get("X-Twilio-Signature"),
        auth_token=settings.TWILIO_AUTH_TOKEN,
    )

    call_sid = _one_field(items, "CallSid")
    called_number = _one_field(items, "To")
    # Require From to be present, but do not use it to resolve tenant identity.
    _one_field(items, "From")

    line = VoiceLineRepository(db).get_by_phone_number(called_number)
    if line is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Voice line not found.",
        )

    try:
        call, _ = VoiceCallRepository(db).get_or_create_call(
            provider_key="twilio",
            provider_call_id=call_sid,
            line=line,
        )
    except VoiceLineNotEligibleError as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Voice line cannot accept this call.",
        ) from exc

    turns = VoiceTurnRepository(db)
    turn = turns.get_for_call_turn(
        tenant_id=line.tenant_id,
        call_id=call.call_id,
        turn_number=1,
    )

    if turn is None:
        # Generate a token only when no initial turn exists. Concurrent
        # duplicate requests may race; the repository enforces uniqueness.
        token = secrets.token_urlsafe(32)
        try:
            turn, _ = turns.create_or_get(
                tenant_id=line.tenant_id,
                call_id=call.call_id,
                turn_number=1,
                callback_token=token,
            )
        except VoiceTurnConflictError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Voice turn conflicts with persisted state.",
            ) from exc

    if turn.status != "awaiting_input":
        # Never reopen a turn that has already been claimed or completed.
        response = VoiceResponse()
        response.say("This call is already being processed. Goodbye.")
        response.hangup()
        return Response(content=str(response), media_type="application/xml")

    try:
        action_url = _speech_action_url(turn.callback_token)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Voice speech callback is not configured.",
        ) from exc

    response = VoiceResponse()
    gather = response.gather(
        input="speech",
        action=action_url,
        method="POST",
        language=line.spoken_language,
        speech_timeout="auto",
        timeout=5,
    )
    gather.say(line.inbound_opening_greeting)
    response.say("I did not hear a response. Please call again. Goodbye.")
    response.hangup()

    return Response(content=str(response), media_type="application/xml")


def _speech_terminal_response(message: str) -> Response:
    response = VoiceResponse()
    response.say(message)
    response.hangup()
    return Response(content=str(response), media_type="application/xml")


@router.post(SPEECH_PATH, status_code=status.HTTP_200_OK)
async def twilio_speech_callback(
    request: Request,
    db: Annotated[Session, Depends(get_db)],
) -> Response:
    """Authenticate and claim one speech turn without persisting its transcript."""
    try:
        configured_url, configured_query = _configured_url(
            settings.TWILIO_SPEECH_WEBHOOK_URL,
            SPEECH_PATH,
        )
        configured_items = parse_qsl(
            configured_query,
            keep_blank_values=True,
            strict_parsing=True,
            max_num_fields=10,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Voice speech callback is not configured.",
        ) from exc

    if any(key == "turn_token" for key, _ in configured_items):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Voice speech callback is not configured.",
        )

    raw_path = request.scope.get("raw_path", b"")
    raw_query = request.scope.get("query_string", b"")

    if raw_path != SPEECH_PATH.encode("ascii"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid voice speech callback.",
        )

    try:
        raw_query_text = raw_query.decode("ascii")
        query_items = parse_qsl(
            raw_query_text,
            keep_blank_values=True,
            strict_parsing=True,
            max_num_fields=11,
        )
    except (UnicodeDecodeError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid voice speech callback query.",
        ) from exc

    token_items = [(key, value) for key, value in query_items if key == "turn_token"]
    if (
        len(token_items) != 1
        or not token_items[0][1]
        or not query_items
        or query_items[-1][0] != "turn_token"
        or query_items[:-1] != configured_items
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid voice speech callback.",
        )

    # Preserve the exact raw query for Twilio signature verification.
    configured_parts = urlsplit(configured_url)
    signed_url = (
        f"{configured_parts.scheme}://{configured_parts.netloc}"
        f"{configured_parts.path}?{raw_query_text}"
    )

    body = await _read_bounded_body(request)
    items = _validate_twilio_request(
        configured_url=signed_url,
        body=body,
        signature=request.headers.get("X-Twilio-Signature"),
        auth_token=settings.TWILIO_AUTH_TOKEN,
    )

    call_sid = _one_field(items, "CallSid")
    called_number = _one_field(items, "To")
    _one_field(items, "From")  # Required, but never used to resolve tenant identity.

    speech_values = [value for key, value in items if key == "SpeechResult"]
    if len(speech_values) > 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid voice speech callback payload.",
        )

    turn = VoiceTurnRepository(db).get_by_callback_token(
        token_items[0][1],
        commit=False,
    )
    if turn is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Voice turn not found.",
        )

    # Resolve ownership through persisted records, never caller-supplied tenant data.
    call = db.get(VoiceCallRecord, turn.call_id)
    line = VoiceLineRepository(db).get_by_phone_number(called_number)

    if (
        call is None
        or line is None
        or turn.tenant_id != call.tenant_id
        or line.tenant_id != call.tenant_id
        or line.id != call.voice_line_id
        or call.provider_key != "twilio"
        or call.provider_call_id != call_sid
        or call.business_line_e164 != called_number
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Voice callback does not match the persisted call.",
        )

    turns = VoiceTurnRepository(db)
    speech = " ".join(speech_values[0].split()) if speech_values else ""

    if not speech:
        try:
            current, _ = turns.mark_no_input(
                tenant_id=turn.tenant_id,
                turn_id=turn.turn_id,
            )
        except VoiceTurnConflictError:
            return _speech_terminal_response("This response has already been processed. Goodbye.")

        if current.status == "no_input":
            return _speech_terminal_response(
                "Sorry, I did not catch that. Please call again. Goodbye."
            )

        return _speech_terminal_response("This response has already been processed. Goodbye.")

    # Claim the turn without persisting a fingerprint of recognized speech.
    try:
        claimed_turn, claimed, claim_token = turns.claim_transient(
            tenant_id=turn.tenant_id,
            turn_id=turn.turn_id,
        )
    except VoiceTurnConflictError:
        return _speech_terminal_response("This response has already been processed. Goodbye.")

    if not claimed or claim_token is None:
        return _speech_terminal_response("This response has already been processed. Goodbye.")

    # Keep execution disabled until the dedicated receptionist-agent
    # mapping and transient-input privacy guarantees are verified.
    turns.fail_transient(
        tenant_id=claimed_turn.tenant_id,
        turn_id=claimed_turn.turn_id,
        claim_token=claim_token,
        failure_code="agent_not_configured",
    )

    return _speech_terminal_response(
        "I'm sorry, I can't process that request right now. Please try again later. Goodbye."
    )
