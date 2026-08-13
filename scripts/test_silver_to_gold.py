from __future__ import annotations

from pathlib import Path

from pyspark.sql import functions as F

from common.spark.spark_builder import SparkSessionBuilder

from spark.transformations.silver_to_gold_transformer import (
    SilverToGoldTransformer,
)

from spark.validation.gold_validator import (
    GoldValidator,
)

from spark.validation.silver_gold_reconciliation import (
    SilverGoldReconciliation,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TEST_DATA = PROJECT_ROOT / "data" / "test" / "silver_vehicle_events_gold_test.csv"


def main() -> None:

    spark = SparkSessionBuilder.build("SilverToGoldDynamicTests")

    try:

        print("=" * 80)
        print("SILVER → GOLD DYNAMIC TESTS")
        print("=" * 80)

        # ---------------------------------------------------------------
        # Load test Silver data
        # ---------------------------------------------------------------

        silver_df = (
            spark.read.option("header", "true").option("inferSchema", "true").csv(str(TEST_DATA))
        )

        silver_df = silver_df.withColumn(
            "event_time",
            F.to_timestamp("event_time"),
        ).withColumn(
            "speed",
            F.col("speed").cast("double"),
        )

        print(f"Test Silver rows : " f"{silver_df.count()}")

        # ---------------------------------------------------------------
        # Run transformer
        # ---------------------------------------------------------------

        gold_df = SilverToGoldTransformer.transform(silver_df)

        gold_df.cache()

        print(f"Test Gold rows : " f"{gold_df.count()}")

        gold_df.show(truncate=False)

        # ---------------------------------------------------------------
        # Structural validation
        # ---------------------------------------------------------------

        GoldValidator.validate(gold_df)

        print("TEST 1 - Gold schema/DQ : PASS")

        # ---------------------------------------------------------------
        # Dynamic expected result
        #
        # Expected result is calculated from the test source,
        # rather than hard-coded as four rows.
        # ---------------------------------------------------------------

        expected_df = (
            silver_df.filter(F.col("vehicle_id").isNotNull())
            .filter(F.col("event_time").isNotNull())
            .filter(F.col("speed").isNotNull())
            .filter(F.col("speed") >= 0)
            .groupBy("vehicle_id")
            .agg(
                F.count("*").cast("long").alias("event_count"),
                F.avg("speed").cast("double").alias("avg_speed"),
                F.min("speed").cast("double").alias("min_speed"),
                F.max("speed").cast("double").alias("max_speed"),
                F.min("event_time").alias("first_event_time"),
                F.max("event_time").alias("last_event_time"),
            )
        )

        # ---------------------------------------------------------------
        # Compare using EXCEPT BOTH in both directions.
        # ---------------------------------------------------------------

        expected_only = expected_df.select(*gold_df.columns).exceptAll(gold_df).count()

        actual_only = gold_df.exceptAll(expected_df.select(*gold_df.columns)).count()

        if expected_only != 0:
            raise AssertionError(f"Expected-only rows: " f"{expected_only}")

        if actual_only != 0:
            raise AssertionError(f"Actual-only rows: " f"{actual_only}")

        print("TEST 2 - Dynamic metric comparison : PASS")

        # ---------------------------------------------------------------
        # Reconciliation
        # ---------------------------------------------------------------

        reconciliation = SilverGoldReconciliation.reconcile(
            silver_df,
            gold_df,
        )

        print("TEST 3 - Silver/Gold reconciliation : PASS")

        print(f"Eligible Silver rows : " f"{reconciliation['eligible_silver_rows']}")

        print(f"Gold rows            : " f"{reconciliation['actual_gold_rows']}")

        print(f"Gold event count     : " f"{reconciliation['actual_event_count']}")

        # ---------------------------------------------------------------
        # Additional semantic assertions
        # ---------------------------------------------------------------

        expected_vehicle_count = silver_df.select("vehicle_id").distinct().count()

        actual_vehicle_count = gold_df.select("vehicle_id").distinct().count()

        if expected_vehicle_count != actual_vehicle_count:
            raise AssertionError("Vehicle cardinality test failed.")

        print("TEST 4 - Vehicle cardinality : PASS")

        # ---------------------------------------------------------------
        # Final
        # ---------------------------------------------------------------

        print("\n" + "=" * 80)
        print("ALL SILVER → GOLD DYNAMIC TESTS PASSED")
        print("=" * 80)

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
