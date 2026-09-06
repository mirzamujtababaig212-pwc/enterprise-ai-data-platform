from unittest.mock import patch

import pytest

from ml.evaluation import (
    EvaluationLineage,
    persist_evaluation_lineage,
)


def test_evaluation_lineage_requires_non_empty_fields():
    with pytest.raises(ValueError, match="dataset_name"):
        EvaluationLineage(
            dataset_name="",
            dataset_version="v1",
            feature_contract_name="vehicle-risk",
            feature_contract_version="v1",
            evaluation_policy_name="vehicle-risk-v1",
        )


def test_evaluation_lineage_as_dict():
    lineage = EvaluationLineage(
        dataset_name="vehicle-risk-training",
        dataset_version="2026-09-01",
        feature_contract_name="vehicle-risk",
        feature_contract_version="v1",
        evaluation_policy_name="vehicle-risk-v1",
    )

    assert lineage.as_dict() == {
        "dataset_name": "vehicle-risk-training",
        "dataset_version": "2026-09-01",
        "feature_contract_name": "vehicle-risk",
        "feature_contract_version": "v1",
        "evaluation_policy_name": "vehicle-risk-v1",
    }


@patch("ml.evaluation.lineage_persistence.mlflow.set_tags")
def test_persist_evaluation_lineage(mock_set_tags):
    lineage = EvaluationLineage(
        dataset_name="vehicle-risk-training",
        dataset_version="2026-09-01",
        feature_contract_name="vehicle-risk",
        feature_contract_version="v1",
        evaluation_policy_name="vehicle-risk-v1",
    )

    persist_evaluation_lineage(lineage)

    mock_set_tags.assert_called_once_with(
        {
            "evaluation_dataset_name": "vehicle-risk-training",
            "evaluation_dataset_version": "2026-09-01",
            "feature_contract_name": "vehicle-risk",
            "feature_contract_version": "v1",
            "evaluation_policy_name": "vehicle-risk-v1",
        }
    )
