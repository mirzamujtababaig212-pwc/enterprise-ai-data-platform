from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class EvaluationRunSummaryResponse(BaseModel):
    run_id: str
    created_at: datetime
    dataset_name: str
    dataset_version: str
    release_passed: bool


class EvaluationRunListResponse(BaseModel):
    runs: list[EvaluationRunSummaryResponse]
    limit: int
    offset: int
    total: int


class EvaluationRunResponse(BaseModel):
    run_id: str
    created_at: datetime
    dataset_name: str
    dataset_version: str
    lineage: dict[str, Any]
    evaluation: dict[str, Any]
    quality_gate: dict[str, Any]
    regression: dict[str, Any] | None
    external_evaluations: list[dict[str, Any]]
    external_quality_gate: dict[str, Any] | None
    passed: bool
    release_passed: bool


class EvaluationComparisonResponse(BaseModel):
    baseline_run_id: str
    candidate_run_id: str
    metrics: dict[str, Any]


class EvaluationReleaseDecisionResponse(BaseModel):
    run_id: str
    passed: bool
    errors: list[str]


class CompositeEvaluationReleaseDecisionResponse(BaseModel):
    run_id: str
    passed: bool
    errors: list[str]
    native: dict[str, Any]
    external: dict[str, Any]


class RetrievalEvaluationPolicyRequest(BaseModel):
    name: str | None = None
    min_recall_at_k: float | None = Field(default=None, ge=0.0, le=1.0)
    min_precision_at_k: float | None = Field(default=None, ge=0.0, le=1.0)
    min_mrr: float | None = Field(default=None, ge=0.0, le=1.0)
    min_ndcg_at_k: float | None = Field(default=None, ge=0.0, le=1.0)
    max_mean_latency_ms: float | None = Field(default=None, ge=0.0)
    min_abstention_accuracy: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )


class RetrievalRegressionPolicyRequest(BaseModel):
    name: str | None = None
    max_recall_at_k_degradation: float = Field(default=0.0, ge=0.0)
    max_precision_at_k_degradation: float = Field(default=0.0, ge=0.0)
    max_mrr_degradation: float = Field(default=0.0, ge=0.0)
    max_ndcg_at_k_degradation: float = Field(default=0.0, ge=0.0)
    max_mean_latency_ms_increase: float = Field(default=0.0, ge=0.0)
    max_abstention_accuracy_degradation: float = Field(default=0.0, ge=0.0)


class EvaluationRunRequest(BaseModel):
    dataset_name: str = Field(min_length=1)
    dataset_version: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    evaluation_policy: RetrievalEvaluationPolicyRequest
    k: int = Field(default=5, ge=1)
    min_relevance_score: float | None = None
    baseline_run_id: str | None = Field(default=None, min_length=1)
    regression_policy: RetrievalRegressionPolicyRequest | None = None


class EvaluationRunExecutionResponse(BaseModel):
    run: EvaluationRunResponse
    release_decision: CompositeEvaluationReleaseDecisionResponse
