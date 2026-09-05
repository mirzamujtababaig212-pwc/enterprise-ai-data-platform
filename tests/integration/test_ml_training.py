from __future__ import annotations

import mlflow.sklearn
import pandas as pd
import pytest

from ai_platform.mlflow.client import MLflowManager
from ml.models.vehicle_risk import (
    FEATURE_COLUMNS,
    MODEL_NAME,
    TARGET_COLUMN,
)
from ml.training import (
    TrainingConfig,
    VehicleRiskTrainer,
)


def _vehicle_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event_count": [
                10,
                11,
                12,
                13,
                14,
                15,
                16,
                17,
                18,
                19,
            ],
            "avg_speed": [
                50,
                55,
                60,
                65,
                70,
                75,
                80,
                85,
                90,
                95,
            ],
            "max_speed": [
                60,
                65,
                70,
                75,
                80,
                115,
                120,
                125,
                130,
                135,
            ],
            "speed_stddev": [
                2,
                2,
                3,
                3,
                4,
                4,
                5,
                5,
                6,
                6,
            ],
            "avg_rpm": [
                1500,
                1550,
                1600,
                1650,
                1700,
                1750,
                1800,
                1850,
                1900,
                1950,
            ],
            "max_rpm": [
                1800,
                1850,
                1900,
                1950,
                2000,
                2050,
                2100,
                2150,
                2200,
                2250,
            ],
            "avg_fuel_level": [
                80,
                78,
                75,
                72,
                70,
                65,
                60,
                55,
                50,
                45,
            ],
            "min_fuel_level": [
                70,
                68,
                65,
                62,
                60,
                55,
                50,
                45,
                40,
                10,
            ],
            "avg_battery": [
                12.5,
                12.5,
                12.4,
                12.4,
                12.3,
                12.3,
                12.2,
                12.2,
                12.1,
                12.1,
            ],
            "avg_engine_temperature": [
                80,
                81,
                82,
                83,
                84,
                85,
                86,
                87,
                88,
                89,
            ],
            "max_engine_temperature": [
                90,
                91,
                92,
                93,
                94,
                95,
                96,
                97,
                98,
                110,
            ],
        }
    )


def test_vehicle_risk_training_end_to_end() -> None:
    trainer = VehicleRiskTrainer()

    result = trainer.train(
        dataframe=_vehicle_dataframe(),
        config=TrainingConfig(
            experiment_name="enterprise-ai-platform",
            run_name="integration-training-test",
            model_params={
                "n_estimators": 25,
                "random_state": 42,
            },
            test_size=0.3,
            random_state=42,
        ),
    )

    assert result.run_id
    assert result.experiment_id
    assert result.model_uri.startswith("runs:/")

    assert result.metadata is not None
    assert result.metadata.model_name == MODEL_NAME
    assert result.metadata.model_type == "RandomForestClassifier"
    assert result.metadata.framework == "scikit-learn"
    assert result.metadata.task_type == "binary_classification"
    assert result.metadata.target_column == TARGET_COLUMN
    assert result.metadata.feature_names == tuple(FEATURE_COLUMNS)
    assert result.metadata.training_run_id == result.run_id
    assert result.metadata.experiment_id == result.experiment_id
    assert result.metadata.model_uri == result.model_uri

    assert "training_accuracy" in result.metrics
    assert "validation_accuracy" in result.metrics

    assert result.training_samples > 0
    assert result.validation_samples > 0

    assert 0.0 <= result.metrics["training_accuracy"] <= 1.0

    assert 0.0 <= result.metrics["validation_accuracy"] <= 1.0

    model = mlflow.sklearn.load_model(result.model_uri)

    dataframe = _vehicle_dataframe()

    predictions = model.predict(dataframe[list(FEATURE_COLUMNS)])

    assert len(predictions) == len(dataframe)

    manager = MLflowManager()

    run = manager.get_run(result.run_id)

    assert run.info.status == "FINISHED"

    assert run.data.metrics["validation_accuracy"] == result.metrics["validation_accuracy"]


def test_vehicle_risk_training_rejects_missing_columns() -> None:
    trainer = VehicleRiskTrainer()

    with pytest.raises(ValueError, match="missing required"):
        trainer.train(
            pd.DataFrame(
                {
                    "event_count": [1, 2],
                }
            )
        )


def test_vehicle_risk_training_rejects_single_class() -> None:
    dataframe = _vehicle_dataframe()

    dataframe["max_speed"] = 50
    dataframe["max_engine_temperature"] = 80
    dataframe["min_fuel_level"] = 50

    trainer = VehicleRiskTrainer()

    with pytest.raises(
        ValueError,
        match="only one target class",
    ):
        trainer.train(dataframe)
