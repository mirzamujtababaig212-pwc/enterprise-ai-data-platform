from __future__ import annotations

from pathlib import Path

from common.config.settings import Settings
from common.logging.logger import get_logger
from common.spark.spark_builder import SparkSessionBuilder
from common.validation.data_quality import DataQualityValidator
from common.writers.delta_writer import DeltaWriter
from spark.transformations.bronze_to_silver_transformer import (
    BronzeToSilverTransformer,
)

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

BRONZE_TABLE = "bronze.vehicle_events"

SILVER_DATABASE = "silver"
SILVER_TABLE = "vehicle_events"
SILVER_FULL_TABLE = f"{SILVER_DATABASE}.{SILVER_TABLE}"

SILVER_PATH = Settings.storage.SILVER_PATH


def main() -> None:

    spark = SparkSessionBuilder.build("BronzeToSilverPipeline")

    try:

        print("=" * 80)
        print("BRONZE → SILVER PIPELINE")
        print("=" * 80)

        # ------------------------------------------------------------------
        # STEP 1: Create Silver database
        # ------------------------------------------------------------------

        spark.sql(
            f"""
            CREATE DATABASE IF NOT EXISTS
            {SILVER_DATABASE}
            """
        )

        logger.info("Starting Bronze to Silver pipeline")

        # ------------------------------------------------------------------
        # STEP 2: Read Bronze
        # ------------------------------------------------------------------

        logger.info(
            "Reading Bronze table=%s",
            BRONZE_TABLE,
        )

        bronze_df = spark.table(BRONZE_TABLE)

        bronze_count = bronze_df.count()

        logger.info(
            "Bronze rows=%s",
            bronze_count,
        )

        print("\nBronze input:")

        bronze_df.show(truncate=False)

        # ------------------------------------------------------------------
        # STEP 3: Transform Bronze → Silver
        # ------------------------------------------------------------------

        silver_df = BronzeToSilverTransformer.transform(bronze_df)

        silver_count = silver_df.count()

        print("\nSilver output:")

        silver_df.show(truncate=False)

        logger.info(
            "Silver rows=%s",
            silver_count,
        )

        # ------------------------------------------------------------------
        # STEP 4: Validate transformation row count
        # ------------------------------------------------------------------

        if silver_count != bronze_count:

            raise RuntimeError(
                "Bronze → Silver transformation changed "
                "the row count unexpectedly: "
                f"Bronze={bronze_count}, "
                f"Silver={silver_count}"
            )

        # ------------------------------------------------------------------
        # STEP 5: Data Quality Validation
        #
        # IMPORTANT:
        # This happens BEFORE the Delta writer.
        # Invalid Silver data must never be written.
        # ------------------------------------------------------------------

        logger.info(
            "Starting data quality validation for %s",
            SILVER_FULL_TABLE,
        )

        validator = DataQualityValidator(
            dataframe=silver_df,
            dataset_name=SILVER_FULL_TABLE,
        )

        validator.check_row_count(minimum=1)

        validator.check_not_null("vehicle_id")

        validator.check_not_null("event_time")

        validator.check_not_null("speed")

        validator.check_min_value(
            "speed",
            0.0,
        )

        validator.check_unique(
            [
                "vehicle_id",
                "event_time",
            ]
        )

        validator.check_columns(
            [
                "vehicle_id",
                "event_time",
                "speed",
            ]
        )

        validator.validate_or_raise()

        logger.info(
            "Data quality validation passed for %s",
            SILVER_FULL_TABLE,
        )

        # ------------------------------------------------------------------
        # STEP 6: Write Silver
        # ------------------------------------------------------------------

        writer = DeltaWriter(
            table=SILVER_FULL_TABLE,
            path=str(SILVER_PATH),
            mode="overwrite",
        )

        writer.write(silver_df)

        # ------------------------------------------------------------------
        # STEP 7: Read the registered Silver table
        # ------------------------------------------------------------------

        print("\nStored Silver table:")

        stored_silver_df = spark.table(SILVER_FULL_TABLE)

        stored_silver_df.show(truncate=False)

        stored_silver_count = stored_silver_df.count()

        logger.info(
            "Stored Silver rows=%s",
            stored_silver_count,
        )

        # ------------------------------------------------------------------
        # STEP 8: Verify row count
        # ------------------------------------------------------------------

        if stored_silver_count != silver_count:

            raise RuntimeError(
                "Silver row-count verification failed: "
                f"expected {silver_count}, "
                f"found {stored_silver_count}"
            )

        # ------------------------------------------------------------------
        # STEP 9: Verify no test data survived
        # ------------------------------------------------------------------

        test_rows = stored_silver_df.filter(stored_silver_df.vehicle_id == "TEST-001").count()

        if test_rows != 0:

            raise RuntimeError("Unexpected TEST-001 record found " "in Silver table.")

        # ------------------------------------------------------------------
        # STEP 10: Display table metadata
        # ------------------------------------------------------------------

        print("\nSilver table metadata:")

        (
            spark.sql(f"DESCRIBE DETAIL {SILVER_FULL_TABLE}")
            .select(
                "format",
                "location",
                "numFiles",
                "sizeInBytes",
            )
            .show(truncate=False)
        )

        # ------------------------------------------------------------------
        # STEP 11: Final success
        # ------------------------------------------------------------------

        print("\n" + "=" * 80)
        print("BRONZE → SILVER PIPELINE " "COMPLETED SUCCESSFULLY")
        print("=" * 80)

        print(f"Bronze rows : {bronze_count}")

        print(f"Silver rows : {stored_silver_count}")

        print(f"Silver table: {SILVER_FULL_TABLE}")

        print(f"Silver path : {SILVER_PATH}")

        print("=" * 80)

        logger.info("Bronze to Silver pipeline " "completed successfully")

    finally:

        spark.stop()


if __name__ == "__main__":
    main()
