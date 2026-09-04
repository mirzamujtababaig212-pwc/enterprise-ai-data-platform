from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class APIError(BaseModel):
    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    details: dict[str, Any] = Field(default_factory=dict)


class APIResponse(BaseModel):
    status: str = Field(min_length=1)
