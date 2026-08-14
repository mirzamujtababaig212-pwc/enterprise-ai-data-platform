from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class GatewayErrorCode(StrEnum):
    AUTHENTICATION_FAILED = "authentication_failed"
    AUTHORIZATION_FAILED = "authorization_failed"
    RATE_LIMITED = "rate_limited"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    MODEL_NOT_FOUND = "model_not_found"
    INVALID_REQUEST = "invalid_request"
    CAPABILITY_NOT_SUPPORTED = "capability_not_supported"
    PROVIDER_ERROR = "provider_error"
    INTERNAL_ERROR = "internal_error"


class GatewayError(BaseModel):
    code: GatewayErrorCode
    message: str
    provider: str | None = None
    model: str | None = None
    retryable: bool = False
    retry_after_seconds: float | None = None
    details: dict[str, Any] = Field(default_factory=dict)
