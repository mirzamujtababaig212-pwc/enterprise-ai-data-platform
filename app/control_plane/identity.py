from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AuthenticatedIdentity:
    """Identity established by control-plane authentication."""

    principal: str
    tenant_id: str
    user_id: str | None = None

    def __post_init__(self) -> None:
        if not self.principal.strip():
            raise ValueError("principal must not be empty")

        if not self.tenant_id.strip():
            raise ValueError("tenant_id must not be empty")

        if self.user_id is not None and not self.user_id.strip():
            raise ValueError("user_id must not be empty when provided")
