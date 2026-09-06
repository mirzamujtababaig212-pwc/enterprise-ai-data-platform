from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelVersionLineage:
    """
    Immutable lineage describing a registered model version and
    the training/evaluation inputs that produced it.
    """

    model_name: str
    model_version: str
    source_run_id: str
    model_uri: str
    dataset_name: str
    dataset_version: str
    feature_contract_name: str
    feature_contract_version: str
    evaluation_policy_name: str

    def __post_init__(self) -> None:
        fields = {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "source_run_id": self.source_run_id,
            "model_uri": self.model_uri,
            "dataset_name": self.dataset_name,
            "dataset_version": self.dataset_version,
            "feature_contract_name": self.feature_contract_name,
            "feature_contract_version": self.feature_contract_version,
            "evaluation_policy_name": self.evaluation_policy_name,
        }

        for name, value in fields.items():
            if not value.strip():
                raise ValueError(f"{name} must not be empty")

    def as_dict(self) -> dict[str, str]:
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "source_run_id": self.source_run_id,
            "model_uri": self.model_uri,
            "dataset_name": self.dataset_name,
            "dataset_version": self.dataset_version,
            "feature_contract_name": self.feature_contract_name,
            "feature_contract_version": self.feature_contract_version,
            "evaluation_policy_name": self.evaluation_policy_name,
        }
