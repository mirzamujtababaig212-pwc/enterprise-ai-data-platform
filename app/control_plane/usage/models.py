from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from pydantic import BaseModel, Field


class UsageEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: str(uuid4()))
    request_id: str = Field(min_length=1)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))

    capability: str = Field(min_length=1)

    provider: str | None = None
    model: str | None = None

    tokens_in: int = Field(default=0, ge=0)
    tokens_out: int = Field(default=0, ge=0)

    estimated_cost: float = Field(default=0.0, ge=0)
    latency_ms: float = Field(default=0, ge=0)

    status: str = Field(min_length=1)
