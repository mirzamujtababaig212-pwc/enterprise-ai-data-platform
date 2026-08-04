from datetime import datetime
from uuid import uuid4

from pydantic import BaseModel, Field


class UsageMetrics(BaseModel):
    request_id: str = Field(default_factory=lambda: str(uuid4()))

    provider: str

    model: str

    prompt_tokens: int

    completion_tokens: int

    total_tokens: int

    latency_ms: float

    cost: float

    timestamp: datetime = Field(default_factory=datetime.utcnow)
