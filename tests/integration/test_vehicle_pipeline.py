from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

from common.logging.logger import get_logger
from common.spark.spark_builder import SparkSessionBuilder

logger = get_logger(__name__)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

BRONZE_TABLE = "bronze.vehicle_events"
SILVER_TABLE = "silver.vehicle_events"
GOLD_TABLE = "gold.vehicle_metrics"


EXPECTED_SILVER_COLUMNS = [
    "vehicle_id",
    "event_time",
    "speed",
]

EXPECTED_GOLD_COLUMNS = [
    "vehicle_id",
    "event_count",
    "avg_speed",
    "min_speed",
    "max_speed",
    "first_event_time",
    "last_event_time",
]


EXPECTED_GOLD = {
    "V001": {
        "event_count": 3,
        "avg_speed": 48.166666666666664,
        "min_speed": 45.5,
        "max_speed": 51.8,
    },
    "V002": {
        "event_count": 3,
        "avg_speed": 34.733333333333334,
        "min_speed": 32.1,
        "max_speed": 36.4,
    },
    "V003": {
        "event_count": 3,
        "avg_speed": 61.5,
        "min_speed": 59.8,
        "max_speed": 63.5,
    },
    "V004": {
        "event_count": 3,
        "avg_speed": 25.03333333333333,
        "min_speed": 22.4,
        "max_speed": 27.1,
    },
}


FLOAT_TOLERANCE = 1e-9


def run_pipeline(
    pipeline_path: Path,
    pipeline_name: str,
) -> None:

    print("\n" + "=" * 80)
    print(f"RUNNING {pipeline_name}")
    print("=" * 80)

    command = [
        sys.executable,
        str(pipeline_path),
    ]

    logger.info(
        "Executing pipeline=%s",
        pipeline_path,
    )

    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
    )

    print("\n--- PIPELINE STDOUT ---")
    print(result.stdout)

    if result.stderr:
        print("\n--- PIPELINE STDERR ---")
        print(result.stderr)

    if result.returncode != 0:

        raise RuntimeError(f"{pipeline_name} failed with exit code " f"{result.returncode}")

    print(f"{pipeline_name} completed successfully.")


def assert_equal(
    actual,
    expected,
    message: str,
) -> None:

    if actual != expected:

        raise AssertionError(f"{message}: " f"expected={expected}, " f"actual={actual}")


def assert_float_equal(
    actual: float,
    expected: float,
    message: str,
) -> None:

    if not math.isclose(
        actual,
        expected,
        rel_tol=FLOAT_TOLERANCE,
        abs_tol=FLOAT_TOLERANCE,
    ):

        raise AssertionError(f"{message}: " f"expected={expected}, " f"actual={actual}")


def validate_silver(
    spark,
) -> None:

    print("\n" + "=" * 80)
    print("VALIDATING SILVER")
    print("=" * 80)

    silver_df = spark.table(SILVER_TABLE)

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    actual_columns = silver_df.columns

    for column in EXPECTED_SILVER_COLUMNS:

        if column not in actual_columns:

            raise AssertionError(f"Silver missing required column: {column}")

    print("Silver required columns: PASS")

    # ------------------------------------------------------------------
    # Row count
    # ------------------------------------------------------------------

    row_count = silver_df.count()

    assert_equal(
        row_count,
        12,
        "Silver row count",
    )

    print("Silver row count = 12: PASS")

    # ------------------------------------------------------------------
    # Null vehicle IDs
    # ------------------------------------------------------------------

    null_vehicle_ids = silver_df.filter(silver_df.vehicle_id.isNull()).count()

    assert_equal(
        null_vehicle_ids,
        0,
        "Silver null vehicle IDs",
    )

    print("Silver null vehicle IDs = 0: PASS")

    # ------------------------------------------------------------------
    # Null event timestamps
    # ------------------------------------------------------------------

    null_event_times = silver_df.filter(silver_df.event_time.isNull()).count()

    assert_equal(
        null_event_times,
        0,
        "Silver null event times",
    )

    print("Silver null event times = 0: PASS")

    # ------------------------------------------------------------------
    # Null speeds
    # ------------------------------------------------------------------

    null_speeds = silver_df.filter(silver_df.speed.isNull()).count()

    assert_equal(
        null_speeds,
        0,
        "Silver null speeds",
    )

    print("Silver null speeds = 0: PASS")

    # ------------------------------------------------------------------
    # Negative speeds
    # ------------------------------------------------------------------

    negative_speeds = silver_df.filter(silver_df.speed < 0).count()

    assert_equal(
        negative_speeds,
        0,
        "Silver negative speeds",
    )

    print("Silver negative speeds = 0: PASS")

    # ------------------------------------------------------------------
    # Duplicate vehicle/time keys
    # ------------------------------------------------------------------

    duplicate_keys = (
        silver_df.groupBy(
            "vehicle_id",
            "event_time",
        )
        .count()
        .filter("count > 1")
        .count()
    )

    assert_equal(
        duplicate_keys,
        0,
        "Silver duplicate vehicle/event-time keys",
    )

    print("Silver duplicate vehicle/event-time keys = 0: PASS")

    # ------------------------------------------------------------------
    # Test data check
    # ------------------------------------------------------------------

    test_rows = silver_df.filter(silver_df.vehicle_id == "TEST-001").count()

    assert_equal(
        test_rows,
        0,
        "Silver TEST-001 rows",
    )

    print("Silver TEST-001 rows = 0: PASS")

    print("\nSILVER VALIDATION: PASSED")


def collect_gold(
    spark,
) -> dict[str, dict[str, float]]:

    gold_df = spark.table(GOLD_TABLE)

    rows = gold_df.orderBy("vehicle_id").collect()

    result: dict[str, dict[str, float]] = {}

    for row in rows:

        result[row["vehicle_id"]] = {
            "event_count": row["event_count"],
            "avg_speed": row["avg_speed"],
            "min_speed": row["min_speed"],
            "max_speed": row["max_speed"],
        }

    return result


def validate_gold(
    spark,
) -> None:

    print("\n" + "=" * 80)
    print("VALIDATING GOLD")
    print("=" * 80)

    gold_df = spark.table(GOLD_TABLE)

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    actual_columns = gold_df.columns

    for column in EXPECTED_GOLD_COLUMNS:

        if column not in actual_columns:

            raise AssertionError(f"Gold missing required column: {column}")

    print("Gold required columns: PASS")

    # ------------------------------------------------------------------
    # Row count
    # ------------------------------------------------------------------

    row_count = gold_df.count()

    assert_equal(
        row_count,
        4,
        "Gold row count",
    )

    print("Gold row count = 4: PASS")

    # ------------------------------------------------------------------
    # Null checks
    # ------------------------------------------------------------------

    for column in EXPECTED_GOLD_COLUMNS:

        null_count = gold_df.filter(gold_df[column].isNull()).count()

        assert_equal(
            null_count,
            0,
            f"Gold null values in {column}",
        )

        print(f"Gold null values in {column} = 0: PASS")

    # ------------------------------------------------------------------
    # Minimum-value checks
    # ------------------------------------------------------------------

    negative_event_counts = gold_df.filter(gold_df.event_count < 1).count()

    assert_equal(
        negative_event_counts,
        0,
        "Gold invalid event_count values",
    )

    print("Gold event_count >= 1: PASS")

    for column in [
        "avg_speed",
        "min_speed",
        "max_speed",
    ]:

        invalid_count = gold_df.filter(gold_df[column] < 0).count()

        assert_equal(
            invalid_count,
            0,
            f"Gold negative {column} values",
        )

        print(f"Gold {column} >= 0: PASS")

    # ------------------------------------------------------------------
    # Unique vehicle IDs
    # ------------------------------------------------------------------

    duplicate_vehicle_ids = gold_df.groupBy("vehicle_id").count().filter("count > 1").count()

    assert_equal(
        duplicate_vehicle_ids,
        0,
        "Gold duplicate vehicle IDs",
    )

    print("Gold duplicate vehicle IDs = 0: PASS")

    # ------------------------------------------------------------------
    # Expected vehicle IDs
    # ------------------------------------------------------------------

    actual_vehicle_ids = {row["vehicle_id"] for row in gold_df.select("vehicle_id").collect()}

    expected_vehicle_ids = set(EXPECTED_GOLD.keys())

    assert_equal(
        actual_vehicle_ids,
        expected_vehicle_ids,
        "Gold vehicle IDs",
    )

    print("Gold vehicle IDs = V001,V002,V003,V004: PASS")

    # ------------------------------------------------------------------
    # Validate metrics
    # ------------------------------------------------------------------

    actual_gold = collect_gold(spark)

    for vehicle_id, expected in EXPECTED_GOLD.items():

        actual = actual_gold[vehicle_id]

        assert_equal(
            actual["event_count"],
            expected["event_count"],
            f"{vehicle_id} event_count",
        )

        assert_float_equal(
            actual["avg_speed"],
            expected["avg_speed"],
            f"{vehicle_id} avg_speed",
        )

        assert_float_equal(
            actual["min_speed"],
            expected["min_speed"],
            f"{vehicle_id} min_speed",
        )

        assert_float_equal(
            actual["max_speed"],
            expected["max_speed"],
            f"{vehicle_id} max_speed",
        )

        print(f"{vehicle_id} metrics: PASS")

    print("\nGOLD VALIDATION: PASSED")


def capture_current_state(
    spark,
) -> dict[str, object]:

    silver_df = spark.table(SILVER_TABLE)

    gold_df = spark.table(GOLD_TABLE)

    silver_rows = [
        (
            row["vehicle_id"],
            row["event_time"],
            row["speed"],
        )
        for row in silver_df.collect()
    ]

    silver_rows = sorted(
        silver_rows,
        key=lambda row: (
            row[0],
            row[1],
            row[2],
        ),
    )

    gold_rows = [
        (
            row["vehicle_id"],
            row["event_count"],
            row["avg_speed"],
            row["min_speed"],
            row["max_speed"],
            row["first_event_time"],
            row["last_event_time"],
        )
        for row in gold_df.collect()
    ]

    gold_rows = sorted(
        gold_rows,
        key=lambda row: row[0],
    )

    return {
        "silver": silver_rows,
        "gold": gold_rows,
    }


def validate_idempotency(
    spark,
) -> None:

    print("\n" + "=" * 80)
    print("VALIDATING IDEMPOTENCY")
    print("=" * 80)

    before = capture_current_state(spark)

    print(
        "Current Silver rows:",
        len(before["silver"]),
    )

    print(
        "Current Gold rows:",
        len(before["gold"]),
    )

    # ------------------------------------------------------------------
    # Run both pipelines again
    # ------------------------------------------------------------------

    run_pipeline(
        PROJECT_ROOT / "spark" / "pipelines" / "bronze_to_silver_pipeline.py",
        "Bronze → Silver Pipeline (Idempotency Run)",
    )

    run_pipeline(
        PROJECT_ROOT / "spark" / "pipelines" / "silver_to_gold_pipeline.py",
        "Silver → Gold Pipeline (Idempotency Run)",
    )

    # ------------------------------------------------------------------
    # Capture state after second execution
    # ------------------------------------------------------------------

    after = capture_current_state(spark)

    # ------------------------------------------------------------------
    # Compare
    # ------------------------------------------------------------------

    if before["silver"] != after["silver"]:

        raise AssertionError("Silver current state changed after " "re-running the pipeline.")

    if before["gold"] != after["gold"]:

        raise AssertionError("Gold current state changed after " "re-running the pipeline.")

    print("Silver current state unchanged: PASS")

    print("Gold current state unchanged: PASS")

    print("IDEMPOTENCY VALIDATION: PASSED")


def main() -> None:

    print("\n" + "=" * 80)
    print("VEHICLE DATA PIPELINE INTEGRATION TEST")
    print("=" * 80)

    # ------------------------------------------------------------------
    # Pipeline paths
    # ------------------------------------------------------------------

    bronze_to_silver_pipeline = (
        PROJECT_ROOT / "spark" / "pipelines" / "bronze_to_silver_pipeline.py"
    )

    silver_to_gold_pipeline = PROJECT_ROOT / "spark" / "pipelines" / "silver_to_gold_pipeline.py"

    # ------------------------------------------------------------------
    # Verify pipeline files exist
    # ------------------------------------------------------------------

    if not bronze_to_silver_pipeline.exists():

        raise FileNotFoundError(f"Pipeline not found: " f"{bronze_to_silver_pipeline}")

    if not silver_to_gold_pipeline.exists():

        raise FileNotFoundError(f"Pipeline not found: " f"{silver_to_gold_pipeline}")

    # ------------------------------------------------------------------
    # STEP 1: Execute Bronze → Silver
    # ------------------------------------------------------------------

    run_pipeline(
        bronze_to_silver_pipeline,
        "Bronze → Silver Pipeline",
    )

    # ------------------------------------------------------------------
    # STEP 2: Execute Silver → Gold
    # ------------------------------------------------------------------

    run_pipeline(
        silver_to_gold_pipeline,
        "Silver → Gold Pipeline",
    )

    # ------------------------------------------------------------------
    # STEP 3: Create validation Spark session
    # ------------------------------------------------------------------

    spark = SparkSessionBuilder.build("VehiclePipelineIntegrationTest")

    try:

        # --------------------------------------------------------------
        # STEP 4: Validate Silver
        # --------------------------------------------------------------

        validate_silver(spark)

        # --------------------------------------------------------------
        # STEP 5: Validate Gold
        # --------------------------------------------------------------

        validate_gold(spark)

        # --------------------------------------------------------------
        # STEP 6: Validate idempotency
        # --------------------------------------------------------------

        validate_idempotency(spark)

        # --------------------------------------------------------------
        # STEP 7: Final validation
        # --------------------------------------------------------------

        print("\n" + "=" * 80)
        print("VEHICLE DATA PIPELINE " "INTEGRATION TEST PASSED")
        print("=" * 80)

        print("Bronze → Silver : PASS")

        print("Silver validation : PASS")

        print("Silver → Gold : PASS")

        print("Gold validation : PASS")

        print("Gold metrics : PASS")

        print("Idempotency : PASS")

        print("=" * 80)

    finally:

        spark.stop()


if __name__ == "__main__":
    main()
