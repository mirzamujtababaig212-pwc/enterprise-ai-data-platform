from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, Sequence, TypeVar

TrainingInputT = TypeVar("TrainingInputT")
TrainingOutputT = TypeVar("TrainingOutputT")

EvaluationInputT = TypeVar("EvaluationInputT")
EvaluationOutputT = TypeVar("EvaluationOutputT")

InferenceInputT = TypeVar("InferenceInputT")
InferenceOutputT = TypeVar("InferenceOutputT")

RegistryVersionT = TypeVar("RegistryVersionT")


class TrainingService(ABC, Generic[TrainingInputT, TrainingOutputT]):
    """
    Framework-agnostic training contract.
    """

    @abstractmethod
    def train(
        self,
        training_input: TrainingInputT,
        config: Any | None = None,
    ) -> TrainingOutputT:
        raise NotImplementedError


class EvaluationService(
    ABC,
    Generic[EvaluationInputT, EvaluationOutputT],
):
    """
    Framework-agnostic model evaluation contract.
    """

    @abstractmethod
    def evaluate(
        self,
        evaluation_input: EvaluationInputT,
    ) -> EvaluationOutputT:
        raise NotImplementedError


class InferenceService(ABC, Generic[InferenceInputT, InferenceOutputT]):
    """
    Framework-agnostic inference contract.
    """

    @abstractmethod
    def predict(
        self,
        inference_input: InferenceInputT,
    ) -> InferenceOutputT:
        raise NotImplementedError

    def predict_batch(
        self,
        inference_input: InferenceInputT,
    ) -> Any:
        raise NotImplementedError("Batch inference is not implemented by this service")


class ModelRegistry(ABC, Generic[RegistryVersionT]):
    """
    Framework-agnostic model registry contract.
    """

    @abstractmethod
    def register_model(
        self,
        model_uri: str,
        model_name: str,
        run_id: str,
        evaluation_passed: bool,
    ) -> RegistryVersionT:
        raise NotImplementedError

    @abstractmethod
    def promote_to_champion(
        self,
        model_name: str,
        version: str,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def rollback_to_version(
        self,
        model_name: str,
        version: str,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_champion_uri(
        self,
        model_name: str,
    ) -> str:
        raise NotImplementedError

    @abstractmethod
    def list_versions(
        self,
        model_name: str,
    ) -> Sequence[RegistryVersionT]:
        raise NotImplementedError
