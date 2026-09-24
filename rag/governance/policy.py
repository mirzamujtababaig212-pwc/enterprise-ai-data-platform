from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class GovernancePolicy:
    """
    Deterministic metadata requirements for governed retrieval.

    Each key/value pair must match the corresponding DocumentChunk
    metadata exactly.
    """

    required_metadata: dict[str, Any] = field(default_factory=dict)
    tenant_id: str | None = None

    def __post_init__(self) -> None:
        for key in self.required_metadata:
            if not isinstance(key, str) or not key.strip():
                raise ValueError("Governance metadata keys must be non-empty strings.")

        if self.tenant_id is not None:
            if not isinstance(self.tenant_id, str):
                raise TypeError("Governance tenant_id must be a string or None.")
            if not self.tenant_id.strip():
                raise ValueError("Governance tenant_id must not be empty.")

            required_tenant_id = self.required_metadata.get("tenant_id")
            if required_tenant_id is not None and required_tenant_id != self.tenant_id:
                raise ValueError(
                    "Governance tenant_id conflicts with " "required_metadata['tenant_id']."
                )

    def to_metadata_filter(self) -> dict[str, Any]:
        """
        Convert this policy into the existing VectorStore metadata filter.
        """

        effective_filter = dict(self.required_metadata)

        if self.tenant_id is not None:
            effective_filter["tenant_id"] = self.tenant_id

        return effective_filter

    def with_tenant_scope(self, tenant_id: str) -> "GovernancePolicy":
        """
        Return this policy constrained to the authenticated tenant.
        """

        if not isinstance(tenant_id, str) or not tenant_id.strip():
            raise ValueError("Governance tenant_id must be a non-empty string.")

        if self.tenant_id is not None and self.tenant_id != tenant_id:
            raise ValueError("Governance policy tenant scope conflicts with authenticated tenant.")

        required_tenant_id = self.required_metadata.get("tenant_id")
        if required_tenant_id is not None and required_tenant_id != tenant_id:
            raise ValueError(
                "Governance policy tenant metadata conflicts with authenticated tenant."
            )

        return GovernancePolicy(
            required_metadata=dict(self.required_metadata),
            tenant_id=tenant_id,
        )
