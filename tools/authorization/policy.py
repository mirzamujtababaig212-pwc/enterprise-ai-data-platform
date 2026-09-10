from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from tools.authorization.models import ToolAuthorizationRequest


@dataclass(frozen=True)
class ToolAuthorizationPolicyResult:
    allowed: bool
    reason: str | None = None


class ToolAuthorizationPolicy(Protocol):
    async def evaluate(
        self,
        request: ToolAuthorizationRequest,
    ) -> ToolAuthorizationPolicyResult: ...


class MetadataAuthorizationPolicy:
    def __init__(
        self,
        required_metadata: dict[str, object],
    ) -> None:
        self._required_metadata = dict(required_metadata)

    async def evaluate(
        self,
        request: ToolAuthorizationRequest,
    ) -> ToolAuthorizationPolicyResult:
        for key, expected_value in self._required_metadata.items():
            actual_value = request.metadata.get(key)

            if actual_value != expected_value:
                return ToolAuthorizationPolicyResult(
                    allowed=False,
                    reason=(
                        f"Authorization metadata requirement failed: " f"{key}={expected_value!r}."
                    ),
                )

        return ToolAuthorizationPolicyResult(
            allowed=True,
            reason="Authorization metadata requirements satisfied.",
        )
