from __future__ import annotations

from typing import Dict

from pyspark.sql import DataFrame
from pyspark.sql import functions as F


class SilverGoldReconciliation:
    """
    Independently reconciles Gold against Silver.

    This validator deliberately does NOT call the Gold transformer
    to calculate the expected result.

    That prevents a bug in the transformer from also appearing in
    the expected dataset.
    """

    @staticmethod
    def _eligible_silver(
        silver_df: DataFrame,
    ) -> DataFrame:

        required = {
            "vehicle_id",
            "event_time",
            "speed",
        }

        missing = required - set(silver_df.columns)

        if missing:
            raise ValueError(
                "Silver is missing columns required for " f"Gold reconciliation: {sorted(missing)}"
            )

        return (
            silver_df.select(
                F.col("vehicle_id").cast("string").alias("vehicle_id"),
                F.to_timestamp(F.col("event_time")).alias("event_time"),
                F.col("speed").cast("double").alias("speed"),
            )
            .filter(F.col("vehicle_id").isNotNull())
            .filter(F.trim(F.col("vehicle_id")) != "")
            .filter(F.col("event_time").isNotNull())
            .filter(F.col("speed").isNotNull())
            .filter(F.col("speed") >= 0)
        )

    @staticmethod
    def build_expected_gold(
        silver_df: DataFrame,
    ) -> DataFrame:

        eligible = SilverGoldReconciliation._eligible_silver(silver_df)

        expected = eligible.groupBy("vehicle_id").agg(
            F.count("*").cast("long").alias("event_count"),
            F.avg("speed").cast("double").alias("avg_speed"),
            F.min("speed").cast("double").alias("min_speed"),
            F.max("speed").cast("double").alias("max_speed"),
            F.min("event_time").alias("first_event_time"),
            F.max("event_time").alias("last_event_time"),
        )

        return expected

    @staticmethod
    def reconcile(
        silver_df: DataFrame,
        gold_df: DataFrame,
    ) -> Dict[str, int]:

        eligible_silver = SilverGoldReconciliation._eligible_silver(silver_df).cache()

        expected_gold = SilverGoldReconciliation.build_expected_gold(silver_df).cache()

        actual_gold = gold_df.cache()

        silver_rows = eligible_silver.count()
        expected_gold_rows = expected_gold.count()
        actual_gold_rows = actual_gold.count()

        silver_distinct_vehicles = eligible_silver.select("vehicle_id").distinct().count()

        gold_event_sum = actual_gold.agg(
            F.coalesce(
                F.sum("event_count"),
                F.lit(0),
            ).cast("long")
        ).first()[0]

        expected_event_sum = expected_gold.agg(
            F.coalesce(
                F.sum("event_count"),
                F.lit(0),
            ).cast("long")
        ).first()[0]

        # ---------------------------------------------------------------
        # Row-count reconciliation
        # ---------------------------------------------------------------

        if actual_gold_rows != expected_gold_rows:
            raise AssertionError(
                "Gold row-count reconciliation failed: "
                f"expected={expected_gold_rows}, "
                f"actual={actual_gold_rows}"
            )

        if actual_gold_rows != silver_distinct_vehicles:
            raise AssertionError(
                "Gold vehicle-grain reconciliation failed: "
                f"distinct Silver vehicles="
                f"{silver_distinct_vehicles}, "
                f"Gold rows={actual_gold_rows}"
            )

        # ---------------------------------------------------------------
        # Event-count conservation
        # ---------------------------------------------------------------

        if gold_event_sum != expected_event_sum:
            raise AssertionError(
                "Gold event-count reconciliation failed: "
                f"expected={expected_event_sum}, "
                f"actual={gold_event_sum}"
            )

        if gold_event_sum != silver_rows:
            raise AssertionError(
                "Silver → Gold event conservation failed: "
                f"eligible Silver rows={silver_rows}, "
                f"Gold event_count sum={gold_event_sum}"
            )

        # ---------------------------------------------------------------
        # Full row-level comparison.
        #
        # Use a tolerance for avg_speed because floating-point
        # aggregation can differ by tiny amounts.
        # ---------------------------------------------------------------

        actual_alias = actual_gold.alias("actual")
        expected_alias = expected_gold.alias("expected")

        joined = expected_alias.join(
            actual_alias,
            on="vehicle_id",
            how="full_outer",
        )

        missing_or_extra = joined.filter(
            F.col("expected.vehicle_id").isNull() | F.col("actual.vehicle_id").isNull()
        ).count()

        if missing_or_extra != 0:
            raise AssertionError(
                "Gold reconciliation found " f"{missing_or_extra} missing/extra vehicle rows."
            )

        tolerance = F.lit(1e-9)

        mismatches = joined.filter(
            (F.col("expected.event_count") != F.col("actual.event_count"))
            | (F.abs(F.col("expected.avg_speed") - F.col("actual.avg_speed")) > tolerance)
            | (F.abs(F.col("expected.min_speed") - F.col("actual.min_speed")) > tolerance)
            | (F.abs(F.col("expected.max_speed") - F.col("actual.max_speed")) > tolerance)
            | (F.col("expected.first_event_time") != F.col("actual.first_event_time"))
            | (F.col("expected.last_event_time") != F.col("actual.last_event_time"))
        ).count()

        if mismatches != 0:
            raise AssertionError(
                "Gold metric reconciliation failed: " f"{mismatches} vehicle rows differ."
            )

        eligible_silver.unpersist()
        expected_gold.unpersist()
        actual_gold.unpersist()

        return {
            "eligible_silver_rows": silver_rows,
            "expected_gold_rows": expected_gold_rows,
            "actual_gold_rows": actual_gold_rows,
            "distinct_silver_vehicles": silver_distinct_vehicles,
            "expected_event_count": int(expected_event_sum),
            "actual_event_count": int(gold_event_sum),
            "mismatched_rows": 0,
        }
