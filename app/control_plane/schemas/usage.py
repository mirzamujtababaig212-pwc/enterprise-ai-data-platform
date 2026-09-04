from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class UsageEventResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    event_id: str
    request_id: str
    timestamp: datetime
    capability: str
    provider: str | None = None
    model: str | None = None
    tokens_in: int = Field(ge=0)
    tokens_out: int = Field(ge=0)
    estimated_cost: float = Field(ge=0)
    latency_ms: float = Field(ge=0)
    status: str


class UsageResponse(BaseModel):
    events: list[UsageEventResponse]
    count: int
