from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ToolAuthorizationRequest:
    principal: str
    tool_name: str


@dataclass(frozen=True)
class ToolAuthorizationResult:
    principal: str
    tool_name: str
    allowed: bool
    reason: str | None = None
