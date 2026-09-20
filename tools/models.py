from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class ToolExecutionFailureCategory(StrEnum):
    TOOL_NOT_FOUND = "tool_not_found"
    TOOL_DISABLED = "tool_disabled"
    SCHEMA_VALIDATION = "schema_validation"
    INVALID_SCHEMA = "invalid_schema"
    MISSING_PRINCIPAL = "missing_principal"
    AUTHORIZATION = "authorization"
    TIMEOUT = "timeout"
    EXECUTION_ERROR = "execution_error"


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    input_schema: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True


@dataclass(frozen=True)
class ToolExecutionResult:
    tool_name: str
    success: bool
    output: Any = None
    error: str | None = None
    failure_category: ToolExecutionFailureCategory | None = None
