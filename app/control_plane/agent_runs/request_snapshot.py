from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from ai_platform.agents.models import AgentRequest
from rag.governance.policy import GovernancePolicy


class AgentRunRequestSnapshot(BaseModel):
    """
    Versioned, durable representation of the request needed to recover
    an interrupted agent run.

    Runtime dependencies such as memory context, LLM clients, tools,
    and observers are intentionally excluded.
    """

    schema_version: int = Field(default=1, ge=1)
    input: str = Field(min_length=1)
    memory_namespace: str | None = None
    governance_policy: dict[str, Any] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_request(cls, request: AgentRequest) -> "AgentRunRequestSnapshot":
        policy = request.governance_policy

        return cls(
            input=request.input,
            memory_namespace=request.memory_namespace,
            governance_policy=(
                {"required_metadata": dict(policy.required_metadata)}
                if policy is not None
                else None
            ),
            metadata=dict(request.metadata),
        )

    def to_request(
        self,
        *,
        session_id: str | None,
        user_id: str | None,
    ) -> AgentRequest:
        policy = self.governance_policy

        governance_policy = (
            GovernancePolicy(required_metadata=dict(policy["required_metadata"]))
            if policy is not None
            else None
        )

        return AgentRequest(
            input=self.input,
            session_id=session_id,
            user_id=user_id,
            memory_namespace=self.memory_namespace,
            governance_policy=governance_policy,
            metadata=dict(self.metadata),
        )
