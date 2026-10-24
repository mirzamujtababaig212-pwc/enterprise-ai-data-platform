from __future__ import annotations

from typing import Annotated
from urllib.parse import urlsplit

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from sqlalchemy.orm import Session

from ai_platform.llm_gateway.config.settings import settings
from app.control_plane.persistence.database import get_db
from app.control_plane.voice.adapters import (
    VoiceWebhookAuthenticationError,
    VoiceWebhookPayloadError,
    authenticate_and_normalize,
)
from app.control_plane.voice.call_repository import (
    VoiceCallRepository,
    VoiceEventConflictError,
    VoiceLineNotEligibleError,
)
from app.control_plane.voice.repository import VoiceLineRepository
from app.control_plane.voice.twilio_adapter import TwilioVoiceStatusAdapter

router = APIRouter(tags=["voice-webhooks"])
CALLBACK_PATH = "/api/v1/voice/webhooks/twilio/status"
MAX_BODY_BYTES = TwilioVoiceStatusAdapter.MAX_BODY_BYTES


def _configured_callback_url() -> tuple[str, str]:
    """Validate callback configuration without disclosing secret values."""
    token = getattr(settings, "TWILIO_AUTH_TOKEN", "")
    callback_url = getattr(settings, "TWILIO_WEBHOOK_URL", "")

    if (
        not isinstance(token, str)
        or not token.strip()
        or not isinstance(callback_url, str)
        or not callback_url.strip()
    ):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Voice webhook is not configured.",
        )

    try:
        parsed = urlsplit(callback_url)
        _ = parsed.port
        callback_url.encode("ascii")
    except (ValueError, UnicodeEncodeError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Voice webhook is not configured.",
        ) from exc

    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
        or parsed.path != CALLBACK_PATH
    ):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Voice webhook is not configured.",
        )

    return callback_url, parsed.query


async def _read_bounded_body(request: Request) -> bytes:
    """Reject oversized requests while streaming, before buffering the full body."""
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


@router.post(CALLBACK_PATH, status_code=status.HTTP_204_NO_CONTENT)
async def twilio_status_callback(
    request: Request,
    db: Annotated[Session, Depends(get_db)],
) -> Response:
    callback_url, configured_query = _configured_callback_url()

    # Never reconstruct the signed URL from Host or forwarded headers.
    # Require the incoming raw path and query to match configuration exactly.
    raw_path = request.scope.get("raw_path", b"")
    raw_query = request.scope.get("query_string", b"")

    if raw_path != CALLBACK_PATH.encode("ascii") or raw_query != configured_query.encode("ascii"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid voice webhook request.",
        )

    body = await _read_bounded_body(request)

    try:
        adapter = TwilioVoiceStatusAdapter(
            auth_token=settings.TWILIO_AUTH_TOKEN,
            webhook_url=callback_url,
        )
        event = authenticate_and_normalize(
            adapter,
            request_url=callback_url,
            headers=request.headers,
            body=body,
            query_params=request.query_params,
        )
    except VoiceWebhookAuthenticationError as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid voice webhook signature.",
        ) from exc
    except VoiceWebhookPayloadError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid voice webhook payload.",
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Voice webhook is not configured.",
        ) from exc

    line = VoiceLineRepository(db).get_by_phone_number(event.business_line_e164)
    if line is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Voice line not found.",
        )

    try:
        # The trusted persisted line supplies tenant identity. The repository
        # handles duplicate delivery and persists only normalized event data.
        VoiceCallRepository(db).process_event(event, line=line)
    except VoiceLineNotEligibleError as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Voice line cannot accept this event.",
        ) from exc
    except VoiceEventConflictError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Voice event conflicts with an existing event.",
        ) from exc

    return Response(status_code=status.HTTP_204_NO_CONTENT)
