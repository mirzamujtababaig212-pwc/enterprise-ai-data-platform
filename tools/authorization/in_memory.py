from __future__ import annotations

from tools.authorization.models import (
    ToolAuthorizationRequest,
    ToolAuthorizationResult,
)
from tools.authorization.policy import ToolAuthorizationPolicy


class InMemoryToolAuthorizer:
    def __init__(
        self,
        policy: ToolAuthorizationPolicy | None = None,
    ) -> None:
        self._permissions: dict[str, set[str]] = {}
        self._policy = policy

    async def allow(
        self,
        principal: str,
        tool_name: str,
    ) -> None:
        if not principal.strip():
            raise ValueError("Principal must not be empty.")

        if not tool_name.strip():
            raise ValueError("Tool name must not be empty.")

        self._permissions.setdefault(
            principal,
            set(),
        ).add(tool_name)

    async def deny(
        self,
        principal: str,
        tool_name: str,
    ) -> None:
        if not principal.strip():
            raise ValueError("Principal must not be empty.")

        if not tool_name.strip():
            raise ValueError("Tool name must not be empty.")

        permissions = self._permissions.get(principal)

        if permissions is None:
            return

        permissions.discard(tool_name)

    async def authorize(
        self,
        request: ToolAuthorizationRequest,
    ) -> ToolAuthorizationResult:
        if not request.principal.strip():
            raise ValueError("Principal must not be empty.")

        if not request.tool_name.strip():
            raise ValueError("Tool name must not be empty.")

        permissions = self._permissions.get(
            request.principal,
            set(),
        )

        if request.tool_name not in permissions:
            return ToolAuthorizationResult(
                principal=request.principal,
                tool_name=request.tool_name,
                allowed=False,
                reason=("Tool is not authorized for this principal."),
            )

        if self._policy is not None:
            policy_result = await self._policy.evaluate(request)

            if not policy_result.allowed:
                return ToolAuthorizationResult(
                    principal=request.principal,
                    tool_name=request.tool_name,
                    allowed=False,
                    reason=(
                        policy_result.reason or "Tool authorization policy denied the request."
                    ),
                )

        return ToolAuthorizationResult(
            principal=request.principal,
            tool_name=request.tool_name,
            allowed=True,
            reason="Tool is authorized.",
        )
