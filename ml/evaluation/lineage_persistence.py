from __future__ import annotations

import mlflow

from .lineage import EvaluationLineage


def persist_evaluation_lineage(lineage: EvaluationLineage) -> None:
    """
    Persist evaluation lineage to the active MLflow run.
    """

    mlflow.set_tags(
        {
            "evaluation_dataset_name": lineage.dataset_name,
            "evaluation_dataset_version": lineage.dataset_version,
            "feature_contract_name": lineage.feature_contract_name,
            "feature_contract_version": lineage.feature_contract_version,
            "evaluation_policy_name": lineage.evaluation_policy_name,
        }
    )
