from __future__ import annotations

from dataclasses import dataclass, field
from typing import FrozenSet


class PolicyViolationError(RuntimeError):
    """Raised when an agent tool execution violates tenant policy."""


@dataclass(frozen=True)
class ModelGovernanceDecision:
    """Immutable model authorization decision captured for an agent run."""

    effective_model: str
    effective_provider: str | None = None
    policy_id: str | None = None
    policy_version: str | None = None

    def __post_init__(self) -> None:
        if not self.effective_model.strip():
            raise ValueError("effective_model must not be empty")

        for field_name, value in (
            ("effective_provider", self.effective_provider),
            ("policy_id", self.policy_id),
            ("policy_version", self.policy_version),
        ):
            if value is not None and not value.strip():
                raise ValueError(f"{field_name} must not be empty when provided")

    def to_dict(self) -> dict[str, str | None]:
        """Serialize the pinned model governance decision."""
        return {
            "effective_model": self.effective_model,
            "effective_provider": self.effective_provider,
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
        }

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, object],
    ) -> "ModelGovernanceDecision":
        """Restore a pinned model governance decision."""
        if not isinstance(payload, dict):
            raise TypeError("Model governance decision payload must be a dictionary.")

        effective_model = payload.get("effective_model")
        effective_provider = payload.get("effective_provider")
        policy_id = payload.get("policy_id")
        policy_version = payload.get("policy_version")

        if not isinstance(effective_model, str):
            raise TypeError("Model governance effective_model must be a string.")

        for field_name, value in (
            ("effective_provider", effective_provider),
            ("policy_id", policy_id),
            ("policy_version", policy_version),
        ):
            if value is not None and not isinstance(value, str):
                raise TypeError(f"Model governance {field_name} must be a string or None.")

        return cls(
            effective_model=effective_model,
            effective_provider=effective_provider,
            policy_id=policy_id,
            policy_version=policy_version,
        )


@dataclass(frozen=True)
class TenantPolicy:
    """Immutable execution policy for a single tenant."""

    tenant_id: str
    allowed_mcp_servers: FrozenSet[str] = field(default_factory=frozenset)
    allowed_tools: FrozenSet[str] = field(default_factory=frozenset)
    blocked_tools: FrozenSet[str] = field(default_factory=frozenset)
    max_tokens_per_run: int | None = None
    allow_cross_tenant_data: bool = False
    allowed_models: FrozenSet[str] | None = None
    allowed_providers: FrozenSet[str] | None = None
    policy_id: str | None = None
    policy_version: str | None = None

    def __post_init__(self) -> None:
        if not self.tenant_id:
            raise ValueError("tenant_id must not be empty")

        if self.max_tokens_per_run is not None and self.max_tokens_per_run < 0:
            raise ValueError("max_tokens_per_run must be non-negative")

        for field_name, values in (
            ("allowed_models", self.allowed_models),
            ("allowed_providers", self.allowed_providers),
        ):
            if values is not None and any(not value.strip() for value in values):
                raise ValueError(f"{field_name} must not contain empty values")

        for field_name, value in (
            ("policy_id", self.policy_id),
            ("policy_version", self.policy_version),
        ):
            if value is not None and not value.strip():
                raise ValueError(f"{field_name} must not be empty when provided")


class TenantPolicyEngine:
    """Evaluates tenant-scoped policy before tool execution."""

    def __init__(self) -> None:
        self._policies: dict[str, TenantPolicy] = {}

    def register_policy(self, policy: TenantPolicy) -> None:
        self._policies[policy.tenant_id] = policy

    def get_policy(self, tenant_id: str) -> TenantPolicy:
        try:
            return self._policies[tenant_id]
        except KeyError as exc:
            raise PolicyViolationError(f"No policy registered for tenant '{tenant_id}'") from exc

    def authorize_model(
        self,
        tenant_id: str,
        model: str,
        provider: str | None = None,
    ) -> ModelGovernanceDecision:
        """
        Authorize the effective model/provider for an agent run.

        Model governance is evaluated once at run admission. The returned
        immutable decision is intended to be pinned to the durable run
        request snapshot so recovery does not silently re-evaluate current
        tenant policy.
        """
        policy = self.get_policy(tenant_id)

        if not isinstance(model, str) or not model.strip():
            raise ValueError("Model must be a non-empty string.")

        normalized_model = model.strip()

        normalized_provider = None
        if provider is not None:
            if not isinstance(provider, str) or not provider.strip():
                raise ValueError("Provider must be a non-empty string when provided.")
            normalized_provider = provider.strip()

        if policy.allowed_models is not None and normalized_model not in policy.allowed_models:
            raise PolicyViolationError(
                f"Model '{normalized_model}' is not allowed " f"for tenant '{tenant_id}'"
            )

        if policy.allowed_providers is not None and (
            normalized_provider is None or normalized_provider not in policy.allowed_providers
        ):
            if normalized_provider is None:
                raise PolicyViolationError(
                    f"No provider was specified for model '{normalized_model}', "
                    f"but tenant '{tenant_id}' has a provider allow-list."
                )

            raise PolicyViolationError(
                f"Provider '{normalized_provider}' is not allowed " f"for tenant '{tenant_id}'"
            )

        return ModelGovernanceDecision(
            effective_model=normalized_model,
            effective_provider=normalized_provider,
            policy_id=policy.policy_id,
            policy_version=policy.policy_version,
        )

    def validate_tool_execution(
        self,
        tenant_id: str,
        tool_name: str,
        server_id: str | None = None,
    ) -> None:
        policy = self.get_policy(tenant_id)

        if tool_name in policy.blocked_tools:
            raise PolicyViolationError(
                f"Tool '{tool_name}' is explicitly blocked " f"for tenant '{tenant_id}'"
            )

        if policy.allowed_tools and tool_name not in policy.allowed_tools:
            raise PolicyViolationError(
                f"Tool '{tool_name}' is not allowed " f"for tenant '{tenant_id}'"
            )

        if server_id and policy.allowed_mcp_servers and server_id not in policy.allowed_mcp_servers:
            raise PolicyViolationError(
                f"MCP server '{server_id}' is not authorized " f"for tenant '{tenant_id}'"
            )
