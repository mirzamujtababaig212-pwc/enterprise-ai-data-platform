from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd

from common.config.settings import Settings
from ml.scripts.train_vehicle_risk_from_silver import (
    build_vehicle_risk_training_dataframe,
    train_vehicle_risk_from_silver,
)


def test_build_vehicle_risk_training_dataframe_reads_canonical_silver_and_materializes_features():
    spark = MagicMock()
    silver = MagicMock()
    features = MagicMock()

    training_dataframe = pd.DataFrame(
        {
            "vehicle_id": ["V001"],
            "event_count": [3],
        }
    )
    features.toPandas.return_value = training_dataframe
    spark.read.format.return_value.load.return_value = silver

    with patch(
        "ml.scripts.train_vehicle_risk_from_silver.VehicleRiskFeatureBuilder.build",
        return_value=features,
    ) as build_features:
        result = build_vehicle_risk_training_dataframe(spark)

    spark.read.format.assert_called_once_with("delta")
    spark.read.format.return_value.load.assert_called_once_with(Settings.storage.SILVER_PATH)
    build_features.assert_called_once_with(silver)
    features.toPandas.assert_called_once_with()
    pd.testing.assert_frame_equal(result, training_dataframe)


def test_train_vehicle_risk_from_silver_passes_materialized_features_to_trainer():
    spark = MagicMock()
    training_dataframe = pd.DataFrame(
        {
            "vehicle_id": ["V001"],
            "event_count": [3],
        }
    )
    config = MagicMock()
    expected_result = MagicMock()

    with (
        patch(
            "ml.scripts.train_vehicle_risk_from_silver.build_vehicle_risk_training_dataframe",
            return_value=training_dataframe,
        ),
        patch("ml.scripts.train_vehicle_risk_from_silver.VehicleRiskTrainer") as trainer_class,
    ):
        trainer_class.return_value.train.return_value = expected_result

        result = train_vehicle_risk_from_silver(spark, config=config)

    trainer_class.assert_called_once_with()
    trainer_class.return_value.train.assert_called_once_with(
        training_dataframe,
        config=config,
    )
    assert result is expected_result
