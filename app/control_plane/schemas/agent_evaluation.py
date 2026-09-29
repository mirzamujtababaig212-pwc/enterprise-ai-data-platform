from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field, model_validator


class AgentEvaluationPolicyRequest(BaseModel):
    policy_id: str | None = None
    policy_version: str | None = None
    max_execution_time_ms: float | None = Field(default=None, ge=0.0)
    max_steps_per_run: int | None = Field(default=None, ge=0)
    max_invalid_tool_calls: int | None = Field(default=None, ge=0)
    allow_governance_denials: bool = False
    require_task_completed: bool = True
    require_answer_match: bool = False
    name: str | None = None

    @model_validator(mode="after")
    def validate_policy_identity_pair(self) -> "AgentEvaluationPolicyRequest":
        if (self.policy_id is None) != (self.policy_version is None):
            raise ValueError("policy_id and policy_version must be provided together.")
        return self


class AgentEvaluationRequest(BaseModel):
    expected_answer: str | None = None
    policy_id: str | None = None
    policy_version: str | None = None
    max_execution_time_ms: float | None = Field(default=None, ge=0.0)
    max_steps_per_run: int | None = Field(default=None, ge=0)
    max_invalid_tool_calls: int | None = Field(default=None, ge=0)
    allow_governance_denials: bool = False
    require_task_completed: bool = True
    require_answer_match: bool = False
    name: str | None = None

    @model_validator(mode="after")
    def validate_policy_identity_pair(self) -> "AgentEvaluationRequest":
        if (self.policy_id is None) != (self.policy_version is None):
            raise ValueError("policy_id and policy_version must be provided together.")
        return self


class AgentEvaluationLineageResponse(BaseModel):
    evaluated_run_id: str
    agent_name: str
    agent_version: str | None = None
    tenant_id: str | None = None
    effective_model: str | None = None
    effective_provider: str | None = None
    model_policy_id: str | None = None
    model_policy_version: str | None = None


class AgentEvaluationMetricsResponse(BaseModel):
    execution_time_ms: float
    steps_total: int
    tool_calls_total: int
    tool_calls_successful: int
    tool_calls_failed: int
    invalid_tool_calls: int
    governance_denials: int
    rag_queries_total: int
    rag_sources_retrieved_total: int
    has_final_answer: bool
    final_answer_length: int
    rag_sources_available_count: int
    rag_unique_chunks_count: int
    grounding_evaluated: bool
    grounding_supported: bool | None
    grounding_support_ratio: float | None
    grounding_supported_sources_total: int
    grounding_source_candidates_total: int
    grounding_method: str | None
    task_completed: bool


class AgentEvaluationQualityGateResponse(BaseModel):
    passed: bool
    violations: list[str]


class AgentAnswerEvaluationResponse(BaseModel):
    evaluated: bool
    exact_match: bool | None
    normalization: str


class AgentEvaluationRunResponse(BaseModel):
    evaluation_run_id: str
    created_at: datetime
    passed: bool
    lineage: AgentEvaluationLineageResponse
    metrics: AgentEvaluationMetricsResponse
    policy: AgentEvaluationPolicyRequest
    quality_gate: AgentEvaluationQualityGateResponse
    answer_evaluation: AgentAnswerEvaluationResponse | None = None


class AgentEvaluationRunListResponse(BaseModel):
    evaluations: list[AgentEvaluationRunResponse]
    limit: int
