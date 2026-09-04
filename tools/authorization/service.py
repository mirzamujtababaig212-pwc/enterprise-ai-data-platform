from __future__ import annotations

from tools.authorization.models import (
    ToolAuthorizationRequest,
    ToolAuthorizationResult,
)
from tools.contracts import ToolAuthorizer


class ToolAuthorizationService:
    def __init__(
        self,
        authorizer: ToolAuthorizer,
    ):
        self.authorizer = authorizer

    async def authorize(
        self,
        principal: str,
        tool_name: str,
    ) -> ToolAuthorizationResult:
        if not principal.strip():
            raise ValueError("Principal must not be empty.")

        if not tool_name.strip():
            raise ValueError("Tool name must not be empty.")

        request = ToolAuthorizationRequest(
            principal=principal,
            tool_name=tool_name,
        )

        return await self.authorizer.authorize(request)
