from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from tools.authorization.models import ToolAuthorizationRequest


@dataclass(frozen=True)
class ToolAuthorizationPolicyResult:
    allowed: bool
    reason: str | None = None
    policy_id: str | None = None
    policy_version: str | None = None


class ToolAuthorizationPolicy(Protocol):
    async def evaluate(
        self,
        request: ToolAuthorizationRequest,
    ) -> ToolAuthorizationPolicyResult: ...


class MetadataAuthorizationPolicy:
    def __init__(
        self,
        required_metadata: dict[str, object],
        *,
        policy_id: str = "metadata_policy",
        policy_version: str = "1.0",
    ) -> None:
        if not policy_id.strip():
            raise ValueError("policy_id must not be empty.")

        if not policy_version.strip():
            raise ValueError("policy_version must not be empty.")

        self._required_metadata = dict(required_metadata)
        self._policy_id = policy_id
        self._policy_version = policy_version

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
                    policy_id=self._policy_id,
                    policy_version=self._policy_version,
                )

        return ToolAuthorizationPolicyResult(
            allowed=True,
            reason="Authorization metadata requirements satisfied.",
            policy_id=self._policy_id,
            policy_version=self._policy_version,
        )
