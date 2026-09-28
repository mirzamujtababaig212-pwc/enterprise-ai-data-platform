from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


@dataclass(frozen=True)
class ToolProvider:
    kind: str
    name: str

    def __post_init__(self) -> None:
        if not isinstance(self.kind, str) or not self.kind.strip():
            raise ValueError("Tool provider kind must not be empty.")

        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Tool provider name must not be empty.")


class ToolExecutionFailureCategory(StrEnum):
    TOOL_NOT_FOUND = "tool_not_found"
    TOOL_DISABLED = "tool_disabled"
    SCHEMA_VALIDATION = "schema_validation"
    INVALID_SCHEMA = "invalid_schema"
    MISSING_PRINCIPAL = "missing_principal"
    TENANT_POLICY = "tenant_policy"
    AUTHORIZATION = "authorization"
    APPROVAL_REJECTED = "approval_rejected"
    EXECUTION_IN_PROGRESS = "execution_in_progress"
    EXECUTION_AMBIGUOUS = "execution_ambiguous"
    TIMEOUT = "timeout"
    EXECUTION_ERROR = "execution_error"


@dataclass(frozen=True)
class ToolExecutionPolicy:
    max_retries: int = 0
    retryable_failure_categories: frozenset[ToolExecutionFailureCategory] = frozenset()
    backoff_seconds: float = 0.0

    def __post_init__(self) -> None:
        if isinstance(self.max_retries, bool) or not isinstance(self.max_retries, int):
            raise TypeError("max_retries must be an integer.")

        if self.max_retries < 0:
            raise ValueError("max_retries must be greater than or equal to zero.")

        if self.backoff_seconds < 0:
            raise ValueError("backoff_seconds must be greater than or equal to zero.")

        for category in self.retryable_failure_categories:
            if not isinstance(category, ToolExecutionFailureCategory):
                raise TypeError(
                    "retryable_failure_categories must contain "
                    "ToolExecutionFailureCategory values."
                )


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    input_schema: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    provider: ToolProvider | None = None
    execution_policy: ToolExecutionPolicy = field(
        default_factory=ToolExecutionPolicy,
    )


@dataclass(frozen=True)
class ToolExecutionResult:
    tool_name: str
    success: bool
    output: Any = None
    error: str | None = None
    failure_category: ToolExecutionFailureCategory | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
