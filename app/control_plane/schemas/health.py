from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(min_length=1)
    service: str = Field(min_length=1)
    version: str = Field(min_length=1)
