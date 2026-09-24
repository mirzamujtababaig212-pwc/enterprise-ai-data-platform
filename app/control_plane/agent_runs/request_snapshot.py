from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from ai_platform.agents.budget import ExecutionBudget
from ai_platform.agents.models import AgentRequest
from ai_platform.agents.policy import ModelGovernanceDecision
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
    principal: str | None = None
    tenant_id: str | None = None
    memory_namespace: str | None = None
    governance_policy: dict[str, Any] | None = None
    model_governance: dict[str, Any] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    execution_budget: dict[str, int | float | None] | None = None

    @classmethod
    def from_request(cls, request: AgentRequest) -> "AgentRunRequestSnapshot":
        policy = request.governance_policy

        return cls(
            input=request.input,
            principal=request.principal,
            tenant_id=request.tenant_id,
            memory_namespace=request.memory_namespace,
            governance_policy=(
                {
                    "required_metadata": dict(policy.required_metadata),
                    "tenant_id": policy.tenant_id,
                }
                if policy is not None
                else None
            ),
            model_governance=(
                request.model_governance.to_dict() if request.model_governance is not None else None
            ),
            metadata=dict(request.metadata),
            execution_budget=(
                {
                    "max_llm_calls": request.execution_budget.max_llm_calls,
                    "max_tool_calls": request.execution_budget.max_tool_calls,
                    "max_tool_rounds": request.execution_budget.max_tool_rounds,
                    "max_duration_seconds": request.execution_budget.max_duration_seconds,
                    "max_tokens_per_run": request.execution_budget.max_tokens_per_run,
                }
                if request.execution_budget is not None
                else None
            ),
        )

    def to_request(
        self,
        *,
        session_id: str | None,
        user_id: str | None,
        principal: str | None,
    ) -> AgentRequest:
        policy = self.governance_policy

        governance_policy = (
            GovernancePolicy(
                required_metadata=dict(policy["required_metadata"]),
                tenant_id=policy.get("tenant_id"),
            )
            if policy is not None
            else None
        )

        execution_budget = (
            ExecutionBudget(**self.execution_budget) if self.execution_budget is not None else None
        )

        model_governance = (
            ModelGovernanceDecision.from_dict(self.model_governance)
            if self.model_governance is not None
            else None
        )

        return AgentRequest(
            input=self.input,
            session_id=session_id,
            user_id=user_id,
            principal=principal,
            tenant_id=self.tenant_id,
            memory_namespace=self.memory_namespace,
            governance_policy=governance_policy,
            model_governance=model_governance,
            metadata=dict(self.metadata),
            execution_budget=execution_budget,
        )
