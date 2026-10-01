from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class FailureDisposition(StrEnum):
    RETRYABLE = "retryable"
    NON_RETRYABLE = "non_retryable"
    AMBIGUOUS = "ambiguous"


class FailureClassification(BaseModel):
    category: str
    disposition: FailureDisposition
    reason: str
    details: dict[str, Any] = Field(default_factory=dict)
