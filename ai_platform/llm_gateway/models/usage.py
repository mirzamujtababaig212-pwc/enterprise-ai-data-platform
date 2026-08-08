from datetime import datetime, timezone
from uuid import uuid4

from pydantic import BaseModel, Field


class UsageMetrics(BaseModel):
    request_id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    latency_ms: float = 0
    tokens_in: int = 0
    tokens_out: int = 0
    estimated_cost: float = 0.0
    status: str = "success"
