from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class EnterpriseEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: str(uuid4()))
    event_type: str
    source: str
    entity_id: str
    event_time: datetime = Field(
        default_factory=lambda: datetime.now(UTC)
    )
    payload: dict[str, Any]
    schema_version: str = "1.0"


def create_event(
    event_type: str,
    source: str,
    entity_id: str,
    payload: dict[str, Any],
) -> EnterpriseEvent:
    return EnterpriseEvent(
        event_type=event_type,
        source=source,
        entity_id=entity_id,
        payload=payload,
    )
