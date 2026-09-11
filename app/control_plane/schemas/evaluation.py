from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel


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
