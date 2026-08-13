#!/usr/bin/env python3

"""
BRONZE -> SILVER QUALITY PIPELINE

Purpose
-------
Reads raw vehicle events from Bronze, safely parses and normalizes
the source payload, classifies data quality, sends invalid records
to Quarantine, and writes valid/recoverable records to Silver.

Important design principle
---------------------------
Malformed source data must NEVER crash the pipeline.

All parsing operations therefore use tolerant/safe parsing.
"""

from pathlib import Path

from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
)

from common.spark.spark_builder import SparkSessionBuilder


# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

BRONZE_PATH = PROJECT_ROOT / "data" / "bronze" / "kafka_vehicle_events"

SILVER_PATH = PROJECT_ROOT / "data" / "silver" / "kafka_vehicle_events"

QUARANTINE_PATH = PROJECT_ROOT / "data" / "quarantine" / "kafka_vehicle_events"


# ============================================================================
# CONFIGURATION
# ============================================================================

APP_NAME = "BronzeToSilverQuality"

SOURCE_FORMAT = "parquet"
TARGET_FORMAT = "delta"

# Required business fields for a Silver vehicle telemetry record.
#
# Validation is currently implemented explicitly in classify_quality()
# because each field has different validation semantics.
REQUIRED_FIELDS = [
    "vehicle_id",
    "event_time",
    "speed",
]


# ============================================================================
# RAW JSON SCHEMA
# ============================================================================
#
# IMPORTANT:
#
# We deliberately parse all business fields as STRING first.
#
# Why?
#
# If we define event_time as TimestampType here, Spark may throw an
# exception when it encounters malformed values such as:
#
#     2026->08-12T12:15:02Z
#
# Instead we parse it as STRING and perform tolerant conversion later.
#
# This ensures malformed source values become NULL and can be classified
# and quarantined rather than terminating the Spark job.
# ============================================================================

RAW_EVENT_SCHEMA = StructType(
    [
        StructField(
            "vehicle_id",
            StringType(),
            True,
        ),
        # Canonical field
        StructField(
            "event_time",
            StringType(),
            True,
        ),
        # Alternate field observed during schema variation
        StructField(
            "timestamp",
            StringType(),
            True,
        ),
        StructField(
            "latitude",
            StringType(),
            True,
        ),
        StructField(
            "longitude",
            StringType(),
            True,
        ),
        StructField(
            "speed",
            StringType(),
            True,
        ),
        StructField(
            "rpm",
            StringType(),
            True,
        ),
        StructField(
            "fuel_level",
            StringType(),
            True,
        ),
        StructField(
            "battery",
            StringType(),
            True,
        ),
        StructField(
            "engine_temperature",
            StringType(),
            True,
        ),
        StructField(
            "gear",
            StringType(),
            True,
        ),
    ]
)


# ============================================================================
# HELPERS
# ============================================================================


def print_header(title: str) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def safe_to_timestamp(column):
    """
    Tolerant timestamp conversion.

    Invalid timestamp values become NULL instead of terminating
    the Spark job.
    """
    return F.col(column).try_cast("timestamp")


def safe_to_double(column):
    """
    Tolerant numeric conversion.

    Invalid numeric values become NULL instead of terminating
    the Spark job.
    """
    return F.col(column).try_cast("double")


def safe_to_int(column):
    """
    Tolerant integer conversion.

    Invalid integer values become NULL instead of terminating
    the Spark job.
    """
    return F.col(column).try_cast("int")


def write_delta(df, path: Path) -> None:
    """
    Write DataFrame as Delta.

    Overwrite is intentional for this batch-oriented local pipeline.
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        df.write.format(TARGET_FORMAT)
        .mode("overwrite")
        .option(
            "overwriteSchema",
            "true",
        )
        .save(str(path))
    )


# ============================================================================
# READ BRONZE
# ============================================================================


def read_bronze(spark):
    print_header("READING BRONZE")

    print(f"Bronze path : {BRONZE_PATH}")

    if not BRONZE_PATH.exists():
        raise FileNotFoundError(f"Bronze path does not exist: {BRONZE_PATH}")

    bronze = spark.read.format(SOURCE_FORMAT).load(str(BRONZE_PATH))

    bronze_count = bronze.count()

    print(f"Bronze rows : {bronze_count}")

    print()
    print("Bronze schema:")

    bronze.printSchema()

    return bronze


# ============================================================================
# PARSE RAW PAYLOAD
# ============================================================================


def parse_raw_payload(bronze):
    """
    Parse raw_value without allowing malformed payloads to crash Spark.

    from_json returns NULL for malformed JSON rather than throwing an
    exception under the normal permissive parsing behavior.
    """

    df = (
        bronze.withColumn(
            "_raw_value_trimmed",
            F.trim(F.col("raw_value")),
        )
        .withColumn(
            "_is_empty_payload",
            (F.col("_raw_value_trimmed").isNull() | (F.col("_raw_value_trimmed") == "")),
        )
        .withColumn(
            "_parsed",
            F.from_json(
                F.col("_raw_value_trimmed"),
                RAW_EVENT_SCHEMA,
            ),
        )
    )

    return df


# ============================================================================
# NORMALIZE
# ============================================================================


def normalize_payload(df):
    """
    Normalize raw JSON into typed Silver columns.

    Timestamp handling:

        event_time
             |
             +---- valid ----------------> event_time
             |
             +---- missing + timestamp --> timestamp
             |
             +---- malformed -----------> quarantine

    All conversions are tolerant.
    """

    df = (
        df
        # ------------------------------------------------------------------
        # Vehicle ID
        # ------------------------------------------------------------------
        .withColumn(
            "_vehicle_id_raw",
            F.col("_parsed.vehicle_id"),
        )
        # ------------------------------------------------------------------
        # Timestamp candidates
        # ------------------------------------------------------------------
        .withColumn(
            "_event_time_raw",
            F.col("_parsed.event_time"),
        )
        .withColumn(
            "_timestamp_raw",
            F.col("_parsed.timestamp"),
        )
        # ------------------------------------------------------------------
        # Safe timestamp conversions
        # ------------------------------------------------------------------
        .withColumn(
            "_event_time_parsed",
            safe_to_timestamp("_event_time_raw"),
        )
        .withColumn(
            "_timestamp_parsed",
            safe_to_timestamp("_timestamp_raw"),
        )
        # ------------------------------------------------------------------
        # Canonical event_time
        #
        # Prefer event_time.
        #
        # If event_time is missing but timestamp is valid,
        # recover using timestamp.
        # ------------------------------------------------------------------
        .withColumn(
            "_event_time",
            F.coalesce(
                F.col("_event_time_parsed"),
                F.col("_timestamp_parsed"),
            ),
        )
        # ------------------------------------------------------------------
        # Numeric fields
        # ------------------------------------------------------------------
        .withColumn(
            "_latitude",
            safe_to_double("_parsed.latitude"),
        )
        .withColumn(
            "_longitude",
            safe_to_double("_parsed.longitude"),
        )
        .withColumn(
            "_speed",
            safe_to_double("_parsed.speed"),
        )
        .withColumn(
            "_rpm",
            safe_to_int("_parsed.rpm"),
        )
        .withColumn(
            "_fuel_level",
            safe_to_double("_parsed.fuel_level"),
        )
        .withColumn(
            "_battery",
            safe_to_double("_parsed.battery"),
        )
        .withColumn(
            "_engine_temperature",
            safe_to_double("_parsed.engine_temperature"),
        )
        .withColumn(
            "_gear",
            safe_to_int("_parsed.gear"),
        )
    )

    return df


# ============================================================================
# DATA QUALITY CLASSIFICATION
# ============================================================================


def classify_quality(df):
    """
    Assign exactly one primary quality classification.

    Categories:

        EMPTY_PAYLOAD
        MALFORMED_JSON
        MISSING_REQUIRED_FIELD
        MALFORMED_TIMESTAMP
        SCHEMA_VARIATION_RECOVERED
        VALID
    """

    # ----------------------------------------------------------------------
    # Malformed JSON
    # ----------------------------------------------------------------------

    parsed_is_null = F.col("_parsed").isNull() & ~F.col("_is_empty_payload")

    # ----------------------------------------------------------------------
    # Required fields
    # ----------------------------------------------------------------------

    vehicle_missing = F.col("_vehicle_id_raw").isNull() | (F.trim(F.col("_vehicle_id_raw")) == "")

    event_time_missing = F.col("_event_time").isNull()

    speed_missing = F.col("_speed").isNull()

    # ----------------------------------------------------------------------
    # Detect malformed canonical event_time specifically.
    #
    # Example:
    #
    #     2026->08-12T12:15:02Z
    #
    # Raw value exists.
    # Parsed timestamp is NULL.
    # ----------------------------------------------------------------------

    malformed_event_time = (
        F.col("_event_time_raw").isNotNull()
        & (F.trim(F.col("_event_time_raw")) != "")
        & F.col("_event_time_parsed").isNull()
    )

    # ----------------------------------------------------------------------
    # Detect malformed alternate timestamp field.
    # ----------------------------------------------------------------------

    malformed_timestamp_alias = (
        F.col("_timestamp_raw").isNotNull()
        & (F.trim(F.col("_timestamp_raw")) != "")
        & F.col("_timestamp_parsed").isNull()
    )

    malformed_timestamp = malformed_event_time | malformed_timestamp_alias

    # ----------------------------------------------------------------------
    # Detect schema variation:
    #
    # timestamp exists
    # event_time does not exist
    # timestamp successfully parses
    # ----------------------------------------------------------------------

    recovered_timestamp = (
        (F.col("_event_time_raw").isNull() | (F.trim(F.col("_event_time_raw")) == ""))
        & F.col("_timestamp_raw").isNotNull()
        & (F.trim(F.col("_timestamp_raw")) != "")
        & F.col("_timestamp_parsed").isNotNull()
    )

    # ----------------------------------------------------------------------
    # Classification priority
    # ----------------------------------------------------------------------

    quality_status = (
        F.when(
            F.col("_is_empty_payload"),
            F.lit("EMPTY_PAYLOAD"),
        )
        .when(
            parsed_is_null,
            F.lit("MALFORMED_JSON"),
        )
        .when(
            malformed_timestamp,
            F.lit("MALFORMED_TIMESTAMP"),
        )
        .when(
            vehicle_missing | event_time_missing | speed_missing,
            F.lit("MISSING_REQUIRED_FIELD"),
        )
        .when(
            recovered_timestamp,
            F.lit("SCHEMA_VARIATION_RECOVERED"),
        )
        .otherwise(
            F.lit("VALID"),
        )
    )

    # ----------------------------------------------------------------------
    # Human-readable reason
    # ----------------------------------------------------------------------

    quality_reason = (
        F.when(
            F.col("_is_empty_payload"),
            F.lit("RAW_PAYLOAD_EMPTY"),
        )
        .when(
            parsed_is_null,
            F.lit("RAW_PAYLOAD_IS_NOT_VALID_JSON"),
        )
        .when(
            malformed_event_time,
            F.lit("EVENT_TIME_IS_MALFORMED"),
        )
        .when(
            malformed_timestamp_alias,
            F.lit("TIMESTAMP_FIELD_IS_MALFORMED"),
        )
        .when(
            recovered_timestamp,
            F.lit("TIMESTAMP_FIELD_NORMALIZED_TO_EVENT_TIME"),
        )
        .when(
            vehicle_missing,
            F.lit("VEHICLE_ID_MISSING"),
        )
        .when(
            event_time_missing,
            F.lit("EVENT_TIME_MISSING"),
        )
        .when(
            speed_missing,
            F.lit("SPEED_MISSING_OR_INVALID"),
        )
        .otherwise(
            F.lit("VALID"),
        )
    )

    df = df.withColumn(
        "quality_status",
        quality_status,
    ).withColumn(
        "quality_reason",
        quality_reason,
    )

    return df


# ============================================================================
# BUSINESS-LEVEL DEDUPLICATION
# ============================================================================


def business_deduplicate(df):
    """
    Perform business-level deduplication.

    Business identity:
        vehicle_id + event_time

    Only records classified as VALID or
    SCHEMA_VARIATION_RECOVERED are eligible.

    Records with missing business keys are not deduplicated here;
    they remain available for quarantine handling.

    For duplicate business keys, retain the most recently ingested
    Kafka record using kafka_timestamp and kafka_offset.
    """

    print_header("BUSINESS-LEVEL DEDUPLICATION")

    valid = (
        df.filter(
            F.col("quality_status").isin(
                "VALID",
                "SCHEMA_VARIATION_RECOVERED",
            )
        )
        .withColumn(
            "_canonical_vehicle_id",
            F.trim(F.col("_vehicle_id_raw")),
        )
        .withColumn(
            "_canonical_event_time",
            F.col("_event_time"),
        )
    )

    dedup_eligible = valid.filter(
        F.col("_canonical_vehicle_id").isNotNull() & F.col("_canonical_event_time").isNotNull()
    )

    dedup_ineligible = valid.filter(
        F.col("_canonical_vehicle_id").isNull() | F.col("_canonical_event_time").isNull()
    )

    window_spec = Window.partitionBy(
        "_canonical_vehicle_id",
        "_canonical_event_time",
    ).orderBy(
        F.col("kafka_timestamp").desc_nulls_last(),
        F.col("kafka_offset").desc_nulls_last(),
    )

    ranked = dedup_eligible.withColumn(
        "_dedup_rank",
        F.row_number().over(window_spec),
    )

    duplicate_count = ranked.filter(F.col("_dedup_rank") > 1).count()

    print(f"Duplicate records removed : {duplicate_count}")

    deduplicated = ranked.filter(F.col("_dedup_rank") == 1).drop("_dedup_rank")

    final_df = deduplicated.unionByName(
        dedup_ineligible,
        allowMissingColumns=True,
    ).drop(
        "_canonical_vehicle_id",
        "_canonical_event_time",
    )

    return (
        final_df,
        duplicate_count,
    )


# ============================================================================
# BUILD SILVER
# ============================================================================


def build_silver(df):
    """
    Build canonical Silver dataset.

    Performs business-level deduplication before projecting
    the canonical Silver schema.
    """

    deduplicated, duplicate_count = business_deduplicate(df)

    silver = deduplicated.filter(
        F.col("quality_status").isin(
            "VALID",
            "SCHEMA_VARIATION_RECOVERED",
        )
    ).select(
        # --------------------------------------------------------------
        # Canonical business fields
        # --------------------------------------------------------------
        F.trim(F.col("_vehicle_id_raw")).alias("vehicle_id"),
        F.col("_event_time").alias("event_time"),
        F.col("_latitude").alias("latitude"),
        F.col("_longitude").alias("longitude"),
        F.col("_speed").alias("speed"),
        F.col("_rpm").alias("rpm"),
        F.col("_fuel_level").alias("fuel_level"),
        F.col("_battery").alias("battery"),
        F.col("_engine_temperature").alias("engine_temperature"),
        F.col("_gear").alias("gear"),
        # --------------------------------------------------------------
        # Kafka lineage
        # --------------------------------------------------------------
        F.col("kafka_key"),
        F.col("kafka_topic"),
        F.col("kafka_partition"),
        F.col("kafka_offset"),
        F.col("kafka_timestamp"),
        # --------------------------------------------------------------
        # Original payload
        # --------------------------------------------------------------
        F.col("raw_value"),
        # --------------------------------------------------------------
        # Operational metadata
        # --------------------------------------------------------------
        F.col("ingestion_time"),
        # --------------------------------------------------------------
        # Quality metadata
        # --------------------------------------------------------------
        F.col("quality_status"),
        F.col("quality_reason"),
    )

    return (
        silver,
        duplicate_count,
    )


# ============================================================================
# BUILD QUARANTINE
# ============================================================================


def build_quarantine(df):
    """
    Build quarantine dataset.

    Every rejected event keeps enough lineage to answer:

        Why was this event rejected?
        What was the original payload?
        Where did it come from?
    """

    quarantine = df.filter(
        ~F.col("quality_status").isin(
            "VALID",
            "SCHEMA_VARIATION_RECOVERED",
        )
    ).select(
        # --------------------------------------------------------------
        # Kafka lineage
        # --------------------------------------------------------------
        F.col("kafka_key"),
        F.col("kafka_topic"),
        F.col("kafka_partition"),
        F.col("kafka_offset"),
        F.col("kafka_timestamp"),
        # --------------------------------------------------------------
        # Original payload
        # --------------------------------------------------------------
        F.col("raw_value"),
        # --------------------------------------------------------------
        # Raw business fields
        # --------------------------------------------------------------
        F.col("_vehicle_id_raw").alias("vehicle_id"),
        F.col("_event_time_raw").alias("event_time_raw"),
        F.col("_timestamp_raw").alias("timestamp_raw"),
        # --------------------------------------------------------------
        # Normalized value, if available
        # --------------------------------------------------------------
        F.col("_event_time").alias("normalized_event_time"),
        # --------------------------------------------------------------
        # Quality classification
        # --------------------------------------------------------------
        F.col("quality_status"),
        F.col("quality_reason"),
        # --------------------------------------------------------------
        # Operational metadata
        # --------------------------------------------------------------
        F.col("ingestion_time"),
        F.current_timestamp().alias("quarantine_time"),
    )

    return quarantine


# ============================================================================
# QUALITY SUMMARY
# ============================================================================


def print_quality_summary(df):
    print_header("QUALITY CLASSIFICATION SUMMARY")

    (
        df.groupBy("quality_status")
        .count()
        .orderBy(F.col("count").desc())
        .show(
            50,
            truncate=False,
        )
    )


# ============================================================================
# PREVIEW QUARANTINE
# ============================================================================


def preview_quarantine(quarantine):
    print_header("QUARANTINE PREVIEW")

    count = quarantine.count()

    print(f"Quarantine rows : {count}")

    if count > 0:
        (
            quarantine.select(
                "kafka_partition",
                "kafka_offset",
                "vehicle_id",
                "event_time_raw",
                "timestamp_raw",
                "quality_status",
                "quality_reason",
                "raw_value",
            )
            .orderBy(
                "kafka_partition",
                "kafka_offset",
            )
            .show(
                50,
                truncate=False,
            )
        )


# ============================================================================
# PREVIEW SILVER
# ============================================================================


def preview_silver(silver):
    print_header("SILVER PREVIEW")

    count = silver.count()

    print(f"Silver rows : {count}")

    (
        silver.select(
            "vehicle_id",
            "event_time",
            "speed",
            "quality_status",
            "quality_reason",
            "kafka_partition",
            "kafka_offset",
        )
        .orderBy(
            "vehicle_id",
            "event_time",
        )
        .show(
            50,
            truncate=False,
        )
    )


# ============================================================================
# MAIN
# ============================================================================


def main():

    print_header("BRONZE → SILVER QUALITY PIPELINE")

    print(f"Bronze path     : {BRONZE_PATH}")

    print(f"Silver path     : {SILVER_PATH}")

    print(f"Quarantine path : {QUARANTINE_PATH}")

    spark = SparkSessionBuilder.build(APP_NAME)

    spark.sparkContext.setLogLevel("WARN")

    try:

        # ------------------------------------------------------------------
        # STEP 1
        # READ BRONZE
        # ------------------------------------------------------------------

        bronze = read_bronze(spark)

        # ------------------------------------------------------------------
        # STEP 2
        # PARSE RAW PAYLOADS
        # ------------------------------------------------------------------

        print_header("PARSING RAW PAYLOADS SAFELY")

        parsed = parse_raw_payload(bronze)

        # ------------------------------------------------------------------
        # STEP 3
        # NORMALIZE PAYLOAD
        # ------------------------------------------------------------------

        print_header("NORMALIZING PAYLOAD")

        normalized = normalize_payload(parsed)

        # ------------------------------------------------------------------
        # STEP 4
        # CLASSIFY DATA QUALITY
        # ------------------------------------------------------------------

        print_header("CLASSIFYING DATA QUALITY")

        classified = classify_quality(normalized)

        # ------------------------------------------------------------------
        # STEP 5
        # QUALITY SUMMARY
        # ------------------------------------------------------------------

        print_quality_summary(classified)

        # ------------------------------------------------------------------
        # STEP 6
        # BUILD QUARANTINE
        # ------------------------------------------------------------------

        quarantine = build_quarantine(classified)

        # ------------------------------------------------------------------
        # STEP 7
        # PREVIEW QUARANTINE
        # ------------------------------------------------------------------

        preview_quarantine(quarantine)

        # ------------------------------------------------------------------
        # STEP 8
        # BUILD SILVER
        # ------------------------------------------------------------------

        silver, duplicate_count = build_silver(classified)

        print()
        print(f"Business duplicate records removed : " f"{duplicate_count}")

        # ------------------------------------------------------------------
        # STEP 9
        # PREVIEW SILVER
        # ------------------------------------------------------------------

        preview_silver(silver)

        # ------------------------------------------------------------------
        # STEP 10
        # WRITE SILVER
        # ------------------------------------------------------------------

        print_header("WRITING SILVER")

        write_delta(
            silver,
            SILVER_PATH,
        )

        print("Silver Delta written to:")

        print(f"  {SILVER_PATH}")

        # ------------------------------------------------------------------
        # STEP 11
        # WRITE QUARANTINE
        # ------------------------------------------------------------------

        print_header("WRITING QUARANTINE")

        write_delta(
            quarantine,
            QUARANTINE_PATH,
        )

        print("Quarantine Delta written to:")

        print(f"  {QUARANTINE_PATH}")

        # ------------------------------------------------------------------
        # STEP 12
        # FINAL COUNTS
        # ------------------------------------------------------------------

        print_header("FINAL COUNTS")

        bronze_count = bronze.count()
        silver_count = silver.count()
        quarantine_count = quarantine.count()

        expected_reconciled_count = silver_count + quarantine_count + duplicate_count

        print(f"Bronze rows                       : " f"{bronze_count}")

        print(f"Silver rows                       : " f"{silver_count}")

        print(f"Quarantine rows                   : " f"{quarantine_count}")

        print(f"Business duplicates removed       : " f"{duplicate_count}")

        print(f"Silver + Quarantine + Duplicates  : " f"{expected_reconciled_count}")

        # ------------------------------------------------------------------
        # STEP 13
        # RECONCILIATION
        # ------------------------------------------------------------------

        if bronze_count != expected_reconciled_count:

            print()
            print("WARNING: Bronze reconciliation failed.")

            print(f"Expected : {bronze_count}")

            print(f"Actual   : {expected_reconciled_count}")

        else:

            print()
            print("RECONCILIATION PASSED")

        # ------------------------------------------------------------------
        # STEP 14
        # COMPLETION
        # ------------------------------------------------------------------

        print_header("BRONZE → SILVER QUALITY PIPELINE " "COMPLETED SUCCESSFULLY")

    except Exception as exc:

        print()

        print("=" * 80)

        print("BRONZE → SILVER PIPELINE FAILED")

        print("=" * 80)

        print(f"Error type : {type(exc).__name__}")

        print(f"Error      : {exc}")

        print("=" * 80)

        raise

    finally:

        spark.stop()


# ============================================================================
# ENTRY POINT
# ============================================================================


if __name__ == "__main__":
    main()
