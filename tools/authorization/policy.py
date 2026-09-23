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


class CompositeToolAuthorizationPolicy:
    """
    Evaluate multiple authorization policies in deterministic order.

    Policy composition is an additional governance gate. Tool permission
    remains the responsibility of the authorizer and is evaluated separately.
    """

    def __init__(
        self,
        policies: list[ToolAuthorizationPolicy] | tuple[ToolAuthorizationPolicy, ...],
        *,
        policy_id: str = "composite_policy",
        policy_version: str = "1.0",
    ) -> None:
        if not policies:
            raise ValueError("At least one authorization policy is required.")

        if not policy_id.strip():
            raise ValueError("policy_id must not be empty.")

        if not policy_version.strip():
            raise ValueError("policy_version must not be empty.")

        self._policies = tuple(policies)
        self._policy_id = policy_id
        self._policy_version = policy_version

    async def evaluate(
        self,
        request: ToolAuthorizationRequest,
    ) -> ToolAuthorizationPolicyResult:
        for policy in self._policies:
            result = await policy.evaluate(request)

            if not result.allowed:
                return result

        return ToolAuthorizationPolicyResult(
            allowed=True,
            reason="All authorization policies satisfied.",
            policy_id=self._policy_id,
            policy_version=self._policy_version,
        )


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


class CapabilityAuthorizationPolicy:
    """
    Authorization policy for Deldai capability metadata.

    Tool permission is evaluated separately by the authorizer. This policy
    evaluates the capability metadata attached to the tool definition.
    """

    def __init__(
        self,
        *,
        allowed_capabilities: set[str] | frozenset[str] | None = None,
        allowed_risk_tiers: set[str] | frozenset[str] | None = None,
        allowed_side_effects: set[bool] | frozenset[bool] | None = None,
        required_permission_scope: str | None = None,
        policy_id: str = "capability_policy",
        policy_version: str = "1.0",
    ) -> None:
        if not policy_id.strip():
            raise ValueError("policy_id must not be empty.")

        if not policy_version.strip():
            raise ValueError("policy_version must not be empty.")

        if required_permission_scope is not None and not required_permission_scope.strip():
            raise ValueError("required_permission_scope must not be empty.")

        self._allowed_capabilities = (
            frozenset(allowed_capabilities) if allowed_capabilities is not None else None
        )
        self._allowed_risk_tiers = (
            frozenset(allowed_risk_tiers) if allowed_risk_tiers is not None else None
        )
        self._allowed_side_effects = (
            frozenset(allowed_side_effects) if allowed_side_effects is not None else None
        )
        self._required_permission_scope = required_permission_scope
        self._policy_id = policy_id
        self._policy_version = policy_version

    async def evaluate(
        self,
        request: ToolAuthorizationRequest,
    ) -> ToolAuthorizationPolicyResult:
        metadata = request.metadata

        if self._allowed_capabilities is not None:
            capability = metadata.get("capability")

            if capability not in self._allowed_capabilities:
                return self._deny(
                    "Authorization capability requirement failed: " f"capability={capability!r}."
                )

        if self._allowed_risk_tiers is not None:
            risk_tier = metadata.get("risk_tier")

            if risk_tier not in self._allowed_risk_tiers:
                return self._deny(
                    "Authorization risk tier requirement failed: " f"risk_tier={risk_tier!r}."
                )

        if self._allowed_side_effects is not None:
            side_effect = metadata.get("side_effect")

            if side_effect not in self._allowed_side_effects:
                return self._deny(
                    "Authorization side-effect requirement failed: " f"side_effect={side_effect!r}."
                )

        if self._required_permission_scope is not None:
            permission_scope = metadata.get("permission_scope")

            if permission_scope != self._required_permission_scope:
                return self._deny(
                    "Authorization permission scope requirement failed: "
                    f"permission_scope={permission_scope!r}."
                )

        return ToolAuthorizationPolicyResult(
            allowed=True,
            reason="Authorization capability requirements satisfied.",
            policy_id=self._policy_id,
            policy_version=self._policy_version,
        )

    def _deny(self, reason: str) -> ToolAuthorizationPolicyResult:
        return ToolAuthorizationPolicyResult(
            allowed=False,
            reason=reason,
            policy_id=self._policy_id,
            policy_version=self._policy_version,
        )
