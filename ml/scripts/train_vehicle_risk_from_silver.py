from __future__ import annotations

import pandas as pd
from pyspark.sql import SparkSession

from common.config.settings import Settings
from common.spark.spark_builder import SparkSessionBuilder
from ml.features.vehicle_risk import VehicleRiskFeatureBuilder
from ml.training import TrainingConfig, TrainingResult, VehicleRiskTrainer


def build_vehicle_risk_training_dataframe(spark: SparkSession) -> pd.DataFrame:
    """Build the Vehicle Risk training dataframe from canonical Silver telemetry."""

    silver = spark.read.format("delta").load(Settings.storage.SILVER_PATH)
    features = VehicleRiskFeatureBuilder.build(silver)
    return features.toPandas()


def train_vehicle_risk_from_silver(
    spark: SparkSession,
    config: TrainingConfig | None = None,
) -> TrainingResult:
    """Train the Vehicle Risk model using features derived from Silver."""

    training_dataframe = build_vehicle_risk_training_dataframe(spark)
    return VehicleRiskTrainer().train(training_dataframe, config=config)


def main() -> TrainingResult:
    """Train Vehicle Risk from the canonical Silver Delta dataset."""

    spark = SparkSessionBuilder.build(app_name="VehicleRiskTrainingFromSilver")
    try:
        return train_vehicle_risk_from_silver(spark)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
