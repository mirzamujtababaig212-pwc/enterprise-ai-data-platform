from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TrainingConfig:
    """
    Configuration for a model-training execution.
    """

    experiment_name: str = "enterprise-ai-platform"

    run_name: str = "training-run"

    model_params: dict[str, Any] = field(default_factory=dict)

    test_size: float = 0.2

    random_state: int = 42

    def __post_init__(self) -> None:
        if not self.experiment_name.strip():
            raise ValueError("experiment_name must not be empty")

        if not self.run_name.strip():
            raise ValueError("run_name must not be empty")

        if not 0.0 < self.test_size < 1.0:
            raise ValueError("test_size must be between 0 and 1")


@dataclass(frozen=True)
class TrainingResult:
    """
    Result returned by a training execution.
    """

    run_id: str

    experiment_id: str

    model_uri: str

    metrics: dict[str, float]

    parameters: dict[str, Any]

    training_samples: int

    validation_samples: int

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must not be empty")

        if not self.experiment_id:
            raise ValueError("experiment_id must not be empty")

        if not self.model_uri:
            raise ValueError("model_uri must not be empty")

        if self.training_samples <= 0:
            raise ValueError("training_samples must be positive")

        if self.validation_samples <= 0:
            raise ValueError("validation_samples must be positive")
