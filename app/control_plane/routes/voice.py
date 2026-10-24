from __future__ import annotations

from collections.abc import Generator
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from sqlalchemy.orm import Session

from app.control_plane.persistence.database import get_db
from app.control_plane.schemas.voice import (
    VoiceLineCreateRequest,
    VoiceLineResponse,
    VoiceLineUpdateRequest,
)
from app.control_plane.voice.repository import (
    DuplicateVoiceLinePhoneNumberError,
    VoiceLineNotFoundError,
    VoiceLineRepository,
)

router = APIRouter(prefix="/api/v1/voice/lines", tags=["voice"])


def _tenant_from_request(request: Request) -> str:
    tenant_id = getattr(request.state, "tenant_id", None)
    principal = getattr(request.state, "principal", None)

    if not tenant_id or not principal:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Authenticated tenant and principal context are required.",
        )

    return tenant_id


def _repository(
    db: Annotated[Session, Depends(get_db)],
) -> Generator[VoiceLineRepository, None, None]:
    yield VoiceLineRepository(db)


RepositoryDep = Annotated[VoiceLineRepository, Depends(_repository)]


@router.get("", response_model=list[VoiceLineResponse])
def list_voice_lines(
    request: Request,
    repository: RepositoryDep,
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> list[VoiceLineResponse]:
    tenant_id = _tenant_from_request(request)
    records = repository.list_for_tenant(
        tenant_id,
        limit=limit,
        offset=offset,
    )
    return [VoiceLineResponse.model_validate(record) for record in records]


@router.post(
    "",
    response_model=VoiceLineResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_voice_line(
    request: Request,
    payload: VoiceLineCreateRequest,
    repository: RepositoryDep,
) -> VoiceLineResponse:
    tenant_id = _tenant_from_request(request)

    try:
        record = repository.create(
            tenant_id=tenant_id,
            values=payload.model_dump(),
        )
    except DuplicateVoiceLinePhoneNumberError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Configured phone number is already assigned.",
        ) from exc

    return VoiceLineResponse.model_validate(record)


@router.get("/{line_id}", response_model=VoiceLineResponse)
def get_voice_line(
    request: Request,
    line_id: int,
    repository: RepositoryDep,
) -> VoiceLineResponse:
    tenant_id = _tenant_from_request(request)
    record = repository.get_for_tenant(line_id, tenant_id)

    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Voice line not found.",
        )

    return VoiceLineResponse.model_validate(record)


@router.patch("/{line_id}", response_model=VoiceLineResponse)
def update_voice_line(
    request: Request,
    line_id: int,
    payload: VoiceLineUpdateRequest,
    repository: RepositoryDep,
) -> VoiceLineResponse:
    tenant_id = _tenant_from_request(request)
    changes = payload.model_dump(exclude_unset=True)

    if not changes:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="At least one voice-line field must be supplied.",
        )

    non_nullable_fields = {
        "configured_phone_number",
        "carrier_trunk_line",
        "inbound_opening_greeting",
        "outbound_opening_message",
        "voice_synthesizer",
        "spoken_language",
        "conversational_pause_pacing",
        "barge_in_sensitivity",
        "llm_reasoning_model",
        "max_call_duration_minutes",
    }
    if any(
        changes.get(field, object()) is None for field in non_nullable_fields if field in changes
    ):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="A non-nullable voice-line field cannot be set to null.",
        )

    try:
        record = repository.update_for_tenant(
            line_id,
            tenant_id=tenant_id,
            changes=changes,
        )
    except VoiceLineNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Voice line not found.",
        ) from exc
    except DuplicateVoiceLinePhoneNumberError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Configured phone number is already assigned.",
        ) from exc

    return VoiceLineResponse.model_validate(record)
