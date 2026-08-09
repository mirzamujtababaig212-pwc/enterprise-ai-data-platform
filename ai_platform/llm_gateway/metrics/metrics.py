from datetime import datetime

from pydantic import BaseModel


class Metrics(BaseModel):
    request_id: str
    timestamp: datetime
    latency_ms: int
    tokens_in: int
    tokens_out: int
    estimated_cost: float
    status: str
