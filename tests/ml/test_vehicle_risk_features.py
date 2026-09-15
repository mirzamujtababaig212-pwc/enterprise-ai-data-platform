import math

import pytest
from pyspark.sql.types import DoubleType, IntegerType, StringType, StructField, StructType

from ml.models.vehicle_risk import FEATURE_COLUMNS
from ml.models.vehicle_risk_features import VEHICLE_RISK_FEATURE_CONTRACT

SILVER_SCHEMA = StructType(
    [
        StructField("vehicle_id", StringType(), True),
        StructField("event_time", StringType(), True),
        StructField("speed", DoubleType(), True),
        StructField("rpm", IntegerType(), True),
        StructField("fuel_level", DoubleType(), True),
        StructField("battery", DoubleType(), True),
        StructField("engine_temperature", DoubleType(), True),
    ]
)


EXPECTED_OUTPUT_COLUMNS = [
    "vehicle_id",
    *FEATURE_COLUMNS,
]


def make_silver_df(spark):
    return spark.createDataFrame(
        [
            ("V001", "2026-01-01 10:00:00", 40.0, 1000, 80.0, 12.5, 80.0),
            ("V001", "2026-01-01 10:01:00", 50.0, 1500, 60.0, 12.0, 90.0),
            ("V001", "2026-01-01 10:02:00", 60.0, 2000, 40.0, 11.5, 100.0),
            ("V002", "2026-01-01 10:00:00", 80.0, 3000, 70.0, 13.0, 95.0),
            ("V002", "2026-01-01 10:01:00", 100.0, 3500, 50.0, 12.5, 105.0),
        ],
        SILVER_SCHEMA,
    )


def test_vehicle_risk_feature_builder_module_exists():
    from ml.features.vehicle_risk import VehicleRiskFeatureBuilder

    assert VehicleRiskFeatureBuilder is not None


def test_vehicle_risk_feature_contract_matches_model_features():
    assert VEHICLE_RISK_FEATURE_CONTRACT.feature_names == FEATURE_COLUMNS


def test_expected_feature_columns_are_model_features_only():
    assert "vehicle_id" not in FEATURE_COLUMNS
    assert len(FEATURE_COLUMNS) == 11


def test_expected_silver_input_columns_are_present():
    expected = {
        "vehicle_id",
        "event_time",
        "speed",
        "rpm",
        "fuel_level",
        "battery",
        "engine_temperature",
    }

    assert expected.issubset(set(SILVER_SCHEMA.fieldNames()))


def test_feature_builder_returns_one_row_per_vehicle(spark):
    from ml.features.vehicle_risk import VehicleRiskFeatureBuilder

    result = VehicleRiskFeatureBuilder.build(make_silver_df(spark))

    assert result.count() == 2
    assert result.select("vehicle_id").distinct().count() == 2


def test_feature_builder_returns_expected_columns(spark):
    from ml.features.vehicle_risk import VehicleRiskFeatureBuilder

    result = VehicleRiskFeatureBuilder.build(make_silver_df(spark))

    assert result.columns == EXPECTED_OUTPUT_COLUMNS


def test_vehicle_risk_feature_aggregations(spark):
    from ml.features.vehicle_risk import VehicleRiskFeatureBuilder

    result = VehicleRiskFeatureBuilder.build(make_silver_df(spark))

    rows = {row["vehicle_id"]: row.asDict() for row in result.collect()}

    v001 = rows["V001"]

    assert v001["event_count"] == 3
    assert v001["avg_speed"] == pytest.approx(50.0)
    assert v001["max_speed"] == pytest.approx(60.0)
    assert v001["speed_stddev"] == pytest.approx(math.sqrt(200.0 / 3.0))
    assert v001["avg_rpm"] == pytest.approx(1500.0)
    assert v001["max_rpm"] == pytest.approx(2000.0)
    assert v001["avg_fuel_level"] == pytest.approx(60.0)
    assert v001["min_fuel_level"] == pytest.approx(40.0)
    assert v001["avg_battery"] == pytest.approx(12.0)
    assert v001["avg_engine_temperature"] == pytest.approx(90.0)
    assert v001["max_engine_temperature"] == pytest.approx(100.0)

    v002 = rows["V002"]

    assert v002["event_count"] == 2
    assert v002["avg_speed"] == pytest.approx(90.0)
    assert v002["max_speed"] == pytest.approx(100.0)
    assert v002["speed_stddev"] == pytest.approx(10.0)
    assert v002["avg_rpm"] == pytest.approx(3250.0)
    assert v002["max_rpm"] == pytest.approx(3500.0)
    assert v002["avg_fuel_level"] == pytest.approx(60.0)
    assert v002["min_fuel_level"] == pytest.approx(50.0)
    assert v002["avg_battery"] == pytest.approx(12.75)
    assert v002["avg_engine_temperature"] == pytest.approx(100.0)
    assert v002["max_engine_temperature"] == pytest.approx(105.0)


def test_speed_stddev_uses_population_standard_deviation(spark):
    from ml.features.vehicle_risk import VehicleRiskFeatureBuilder

    df = spark.createDataFrame(
        [
            ("V001", "2026-01-01 10:00:00", 40.0, 1000, 80.0, 12.5, 80.0),
            ("V001", "2026-01-01 10:01:00", 50.0, 1000, 80.0, 12.5, 80.0),
            ("V001", "2026-01-01 10:02:00", 60.0, 1000, 80.0, 12.5, 80.0),
        ],
        SILVER_SCHEMA,
    )

    row = VehicleRiskFeatureBuilder.build(df).collect()[0]

    assert row["speed_stddev"] == pytest.approx(math.sqrt(200.0 / 3.0))


def test_feature_builder_preserves_zero_speed_and_zero_fuel(spark):
    from ml.features.vehicle_risk import VehicleRiskFeatureBuilder

    df = spark.createDataFrame(
        [
            ("V001", "2026-01-01 10:00:00", 0.0, 1000, 0.0, 0.0, 80.0),
            ("V001", "2026-01-01 10:01:00", 10.0, 1200, 20.0, 10.0, 90.0),
        ],
        SILVER_SCHEMA,
    )

    row = VehicleRiskFeatureBuilder.build(df).collect()[0]

    assert row["event_count"] == 2
    assert row["min_fuel_level"] == pytest.approx(0.0)
    assert row["avg_speed"] == pytest.approx(5.0)


def test_feature_builder_rejects_missing_required_columns(spark):
    from ml.features.vehicle_risk import VehicleRiskFeatureBuilder

    df = spark.createDataFrame(
        [
            ("V001", "2026-01-01 10:00:00", 40.0),
        ],
        [
            "vehicle_id",
            "event_time",
            "speed",
        ],
    )

    with pytest.raises(ValueError, match="missing required columns"):
        VehicleRiskFeatureBuilder.build(df)


def test_feature_builder_rejects_none_input():
    from ml.features.vehicle_risk import VehicleRiskFeatureBuilder

    with pytest.raises(ValueError, match="cannot be None"):
        VehicleRiskFeatureBuilder.build(None)


def test_feature_builder_empty_input_returns_empty_result(spark):
    from ml.features.vehicle_risk import VehicleRiskFeatureBuilder

    empty = spark.createDataFrame([], SILVER_SCHEMA)

    result = VehicleRiskFeatureBuilder.build(empty)

    assert result.count() == 0
    assert result.columns == EXPECTED_OUTPUT_COLUMNS


def test_feature_builder_excludes_invalid_vehicle_records(spark):
    from ml.features.vehicle_risk import VehicleRiskFeatureBuilder

    df = spark.createDataFrame(
        [
            (None, "2026-01-01 10:00:00", 40.0, 1000, 80.0, 12.5, 80.0),
            ("", "2026-01-01 10:01:00", 50.0, 1000, 80.0, 12.5, 80.0),
            ("   ", "2026-01-01 10:02:00", 60.0, 1000, 80.0, 12.5, 80.0),
            ("V001", None, 70.0, 1000, 80.0, 12.5, 80.0),
            ("V001", "2026-01-01 10:04:00", 80.0, 1000, 80.0, 12.5, 80.0),
        ],
        SILVER_SCHEMA,
    )

    result = VehicleRiskFeatureBuilder.build(df)

    rows = result.collect()

    assert len(rows) == 1
    assert rows[0]["vehicle_id"] == "V001"
    assert rows[0]["event_count"] == 1


def test_feature_builder_output_types_match_feature_contract(spark):
    from ml.features.vehicle_risk import VehicleRiskFeatureBuilder

    result = VehicleRiskFeatureBuilder.build(make_silver_df(spark))

    fields = {field.name: field.dataType for field in result.schema.fields}

    assert isinstance(fields["event_count"], IntegerType)
    assert isinstance(fields["avg_speed"], DoubleType)
    assert isinstance(fields["max_speed"], DoubleType)
    assert isinstance(fields["speed_stddev"], DoubleType)
    assert isinstance(fields["avg_rpm"], DoubleType)
    assert isinstance(fields["max_rpm"], DoubleType)
    assert isinstance(fields["avg_fuel_level"], DoubleType)
    assert isinstance(fields["min_fuel_level"], DoubleType)
    assert isinstance(fields["avg_battery"], DoubleType)
    assert isinstance(fields["avg_engine_temperature"], DoubleType)
    assert isinstance(fields["max_engine_temperature"], DoubleType)


def test_bootstrap_risk_labels_are_compatible_with_feature_output(spark):
    from ml.features.vehicle_risk import VehicleRiskFeatureBuilder
    from ml.training.vehicle_risk import VehicleRiskTrainer

    df = spark.createDataFrame(
        [
            ("V001", "2026-01-01 10:00:00", 40.0, 1000, 80.0, 12.5, 80.0),
            ("V001", "2026-01-01 10:01:00", 50.0, 1000, 80.0, 12.5, 90.0),
            ("V001", "2026-01-01 10:02:00", 60.0, 1000, 80.0, 12.5, 100.0),
            ("V002", "2026-01-01 10:00:00", 120.0, 1000, 80.0, 12.5, 90.0),
        ],
        SILVER_SCHEMA,
    )

    features = VehicleRiskFeatureBuilder.build(df).toPandas()
    labelled = VehicleRiskTrainer._create_bootstrap_labels(features)

    rows = {row["vehicle_id"]: row["risk"] for _, row in labelled.iterrows()}

    assert rows["V001"] == 0
    assert rows["V002"] == 1
