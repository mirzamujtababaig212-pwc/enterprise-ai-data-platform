from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ModelMetadata:
    """
    Platform-level metadata describing a trained ML model.

    This object is deliberately framework-agnostic so the same
    lifecycle can support scikit-learn, PyTorch, XGBoost, and
    other model implementations.
    """

    model_name: str
    model_version: str | None = None
    model_type: str = ""
    framework: str = ""
    task_type: str = ""
    target_column: str | None = None
    feature_names: tuple[str, ...] = ()
    training_run_id: str | None = None
    experiment_id: str | None = None
    model_uri: str | None = None
    model_alias: str | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    parameters: dict[str, Any] = field(default_factory=dict)
    tags: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.model_name.strip():
            raise ValueError("model_name must not be empty")

        if self.model_version is not None and not self.model_version.strip():
            raise ValueError("model_version must not be empty")

        if self.model_type and not self.model_type.strip():
            raise ValueError("model_type must not be whitespace")

        if self.framework and not self.framework.strip():
            raise ValueError("framework must not be whitespace")

        if self.task_type and not self.task_type.strip():
            raise ValueError("task_type must not be whitespace")

        object.__setattr__(self, "feature_names", tuple(self.feature_names))
        object.__setattr__(self, "metrics", dict(self.metrics))
        object.__setattr__(self, "parameters", dict(self.parameters))
        object.__setattr__(self, "tags", dict(self.tags))
