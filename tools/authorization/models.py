from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ToolAuthorizationRequest:
    principal: str
    tool_name: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolAuthorizationResult:
    principal: str
    tool_name: str
    allowed: bool
    reason: str | None = None
    policy_id: str | None = None
    policy_version: str | None = None
