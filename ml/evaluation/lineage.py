from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvaluationLineage:
    """
    Immutable lineage describing the inputs and governance context
    associated with a model evaluation.
    """

    dataset_name: str
    dataset_version: str
    feature_contract_name: str
    feature_contract_version: str
    evaluation_policy_name: str

    def __post_init__(self) -> None:
        fields = {
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
            "dataset_name": self.dataset_name,
            "dataset_version": self.dataset_version,
            "feature_contract_name": self.feature_contract_name,
            "feature_contract_version": self.feature_contract_version,
            "evaluation_policy_name": self.evaluation_policy_name,
        }
