from __future__ import annotations

from typing import Protocol


class ApprovalOverrideAuthorizer(Protocol):
    """Authorize principals to perform approval overrides."""

    def is_authorized(self, principal: str) -> bool: ...


class ConfiguredApprovalOverrideAuthorizer:
    """Authorize approval overrides for explicitly configured principals."""

    def __init__(self, principals: frozenset[str]) -> None:
        self._principals = principals

    def is_authorized(self, principal: str) -> bool:
        if not principal.strip():
            raise ValueError("principal must not be empty")

        return principal in self._principals
