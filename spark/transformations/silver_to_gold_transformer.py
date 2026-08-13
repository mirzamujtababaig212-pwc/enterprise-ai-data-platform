from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from spark.schemas.gold_schema import GOLD_REQUIRED_COLUMNS


class SilverToGoldTransformer:
    """
    Production Silver -> Gold transformation.

    Input:
        silver.vehicle_events

    Input grain:
        one vehicle event per row

    Output:
        one aggregate row per vehicle_id

    Gold metrics:
        - event_count
        - avg_speed
        - min_speed
        - max_speed
        - first_event_time
        - last_event_time

    The transformer intentionally does NOT contain any hard-coded
    expected row counts.
    """

    REQUIRED_SILVER_COLUMNS = [
        "vehicle_id",
        "event_time",
        "speed",
    ]

    @staticmethod
    def transform(df: DataFrame) -> DataFrame:
        """
        Transform Silver vehicle events into vehicle-level Gold metrics.
        """

        if df is None:
            raise ValueError("Silver DataFrame cannot be None.")

        missing_columns = set(SilverToGoldTransformer.REQUIRED_SILVER_COLUMNS) - set(df.columns)

        if missing_columns:
            raise ValueError(
                "Silver DataFrame is missing required columns: " f"{sorted(missing_columns)}"
            )

        # ---------------------------------------------------------------
        # Step 1: Select only the columns required by the Gold contract.
        # ---------------------------------------------------------------

        cleaned_df = df.select(
            F.col("vehicle_id").cast("string").alias("vehicle_id"),
            F.to_timestamp(F.col("event_time")).alias("event_time"),
            F.col("speed").cast("double").alias("speed"),
        )

        # ---------------------------------------------------------------
        # Step 2: Remove records that cannot participate in Gold metrics.
        #
        # Silver should already have removed these, but Gold must still
        # protect its own contract.
        # ---------------------------------------------------------------

        cleaned_df = (
            cleaned_df.filter(F.col("vehicle_id").isNotNull())
            .filter(F.trim(F.col("vehicle_id")) != "")
            .filter(F.col("event_time").isNotNull())
            .filter(F.col("speed").isNotNull())
            .filter(F.col("speed") >= F.lit(0.0))
        )

        # ---------------------------------------------------------------
        # Step 3: Aggregate to Gold grain:
        #
        #       one row per vehicle_id
        # ---------------------------------------------------------------

        gold_df = cleaned_df.groupBy("vehicle_id").agg(
            F.count("*").cast("long").alias("event_count"),
            F.avg("speed").cast("double").alias("avg_speed"),
            F.min("speed").cast("double").alias("min_speed"),
            F.max("speed").cast("double").alias("max_speed"),
            F.min("event_time").alias("first_event_time"),
            F.max("event_time").alias("last_event_time"),
        )

        # ---------------------------------------------------------------
        # Step 4: Explicitly select the Gold contract.
        # ---------------------------------------------------------------

        gold_df = gold_df.select(*GOLD_REQUIRED_COLUMNS)

        # ---------------------------------------------------------------
        # Step 5: Deterministic ordering for previews/tests.
        # ---------------------------------------------------------------

        gold_df = gold_df.orderBy(F.col("vehicle_id"))

        return gold_df
