from __future__ import annotations

from pydantic import BaseModel, Field


class Capability(BaseModel):
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    status: str = Field(min_length=1)


class CapabilitiesResponse(BaseModel):
    service: str = Field(min_length=1)
    capabilities: list[Capability]
