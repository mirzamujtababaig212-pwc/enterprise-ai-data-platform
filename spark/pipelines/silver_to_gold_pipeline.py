from __future__ import annotations
from common.config.settings import Settings
from pathlib import Path

from common.logging.logger import get_logger
from common.spark.spark_builder import SparkSessionBuilder
from common.validation.data_quality import DataQualityValidator
from common.writers.delta_writer import DeltaWriter

from spark.schemas.gold_schema import (
    GOLD_REQUIRED_COLUMNS,
)
from spark.transformations.silver_to_gold_transformer import (
    SilverToGoldTransformer,
)
from spark.validation.gold_validator import (
    GoldValidator,
)
from spark.validation.silver_gold_reconciliation import (
    SilverGoldReconciliation,
)


logger = get_logger(__name__)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


SILVER_DATABASE = "silver"
SILVER_TABLE = "vehicle_events"
SILVER_FULL_TABLE = f"{SILVER_DATABASE}.{SILVER_TABLE}"

GOLD_DATABASE = "gold"
GOLD_TABLE = "vehicle_metrics"
GOLD_FULL_TABLE = f"{GOLD_DATABASE}.{GOLD_TABLE}"


SILVER_PATH = Settings.storage.SILVER_PATH
GOLD_PATH = Settings.storage.GOLD_PATH


def main() -> None:

    spark = SparkSessionBuilder.build("SilverToGoldPipeline")

    try:

        print("=" * 80)
        print("SILVER → GOLD PRODUCTION PIPELINE")
        print("=" * 80)

        # ---------------------------------------------------------------
        # STEP 1: Ensure Gold database exists
        # ---------------------------------------------------------------

        spark.sql(f"CREATE DATABASE IF NOT EXISTS " f"{GOLD_DATABASE}")

        logger.info("Starting Silver → Gold pipeline")

        # ---------------------------------------------------------------
        # STEP 2: Read Silver
        # ---------------------------------------------------------------

        print("\n" + "=" * 80)
        print("READING SILVER")
        print("=" * 80)

        logger.info(
            "Reading Silver table=%s",
            SILVER_FULL_TABLE,
        )

        silver_df = spark.table(SILVER_FULL_TABLE).cache()

        silver_count = silver_df.count()

        print(f"Silver rows : {silver_count}")

        if silver_count == 0:
            raise RuntimeError("Silver dataset is empty. " "Gold pipeline will not run.")

        # ---------------------------------------------------------------
        # STEP 3: Transform Silver → Gold
        # ---------------------------------------------------------------

        print("\n" + "=" * 80)
        print("TRANSFORMING SILVER → GOLD")
        print("=" * 80)

        gold_df = SilverToGoldTransformer.transform(silver_df).cache()

        gold_count = gold_df.count()

        print(f"Gold rows : {gold_count}")

        if gold_count == 0:
            raise RuntimeError("Gold transformation produced zero rows.")

        print("\nGold preview:")
        gold_df.show(
            20,
            truncate=False,
        )

        # ---------------------------------------------------------------
        # STEP 4: Generic Data Quality Validation
        # ---------------------------------------------------------------

        print("\n" + "=" * 80)
        print("GOLD DATA QUALITY VALIDATION")
        print("=" * 80)

        validator = DataQualityValidator(
            dataframe=gold_df,
            dataset_name=GOLD_FULL_TABLE,
        )

        validator.check_row_count(minimum=1)

        validator.check_not_null("vehicle_id")

        validator.check_not_null("event_count")

        validator.check_not_null("avg_speed")

        validator.check_not_null("min_speed")

        validator.check_not_null("max_speed")

        validator.check_not_null("first_event_time")

        validator.check_not_null("last_event_time")

        validator.check_min_value(
            "event_count",
            1,
        )

        validator.check_min_value(
            "avg_speed",
            0.0,
        )

        validator.check_min_value(
            "min_speed",
            0.0,
        )

        validator.check_min_value(
            "max_speed",
            0.0,
        )

        validator.check_unique(["vehicle_id"])

        validator.check_columns(GOLD_REQUIRED_COLUMNS)

        validator.validate_or_raise()

        print("Generic Gold DQ validation : PASS")

        # ---------------------------------------------------------------
        # STEP 5: Gold-specific semantic validation
        # ---------------------------------------------------------------

        print("\n" + "=" * 80)
        print("GOLD SEMANTIC VALIDATION")
        print("=" * 80)

        GoldValidator.validate_schema(gold_df)

        GoldValidator.validate_not_null(gold_df)

        GoldValidator.validate_numeric_ranges(gold_df)

        GoldValidator.validate_speed_order(gold_df)

        GoldValidator.validate_time_order(gold_df)

        GoldValidator.validate_unique_vehicle(gold_df)

        print("Gold semantic validation : PASS")

        # ---------------------------------------------------------------
        # STEP 6: Independent Silver → Gold reconciliation
        # ---------------------------------------------------------------

        print("\n" + "=" * 80)
        print("SILVER → GOLD RECONCILIATION")
        print("=" * 80)

        reconciliation = SilverGoldReconciliation.reconcile(
            silver_df=silver_df,
            gold_df=gold_df,
        )

        print("Eligible Silver rows       : " f"{reconciliation['eligible_silver_rows']}")

        print("Expected Gold rows         : " f"{reconciliation['expected_gold_rows']}")

        print("Actual Gold rows           : " f"{reconciliation['actual_gold_rows']}")

        print("Distinct Silver vehicles  : " f"{reconciliation['distinct_silver_vehicles']}")

        print("Expected event count      : " f"{reconciliation['expected_event_count']}")

        print("Actual Gold event count   : " f"{reconciliation['actual_event_count']}")

        print("Metric mismatches         : " f"{reconciliation['mismatched_rows']}")

        print("\nSilver → Gold reconciliation : PASS")

        # ---------------------------------------------------------------
        # STEP 7: Write Gold
        # ---------------------------------------------------------------

        print("\n" + "=" * 80)
        print("WRITING GOLD")
        print("=" * 80)

        writer = DeltaWriter(
            table=GOLD_FULL_TABLE,
            path=str(GOLD_PATH),
            mode="overwrite",
        )

        writer.write(gold_df)

        print(f"Gold Delta written to:\n" f"  {GOLD_PATH}")

        # ---------------------------------------------------------------
        # STEP 8: Read Gold back
        # ---------------------------------------------------------------

        print("\n" + "=" * 80)
        print("READING GOLD BACK")
        print("=" * 80)

        stored_gold_df = spark.table(GOLD_FULL_TABLE).cache()

        stored_gold_count = stored_gold_df.count()

        print(f"Stored Gold rows : " f"{stored_gold_count}")

        stored_gold_df.show(
            20,
            truncate=False,
        )

        # ---------------------------------------------------------------
        # STEP 9: Storage reconciliation
        # ---------------------------------------------------------------

        if stored_gold_count != gold_count:
            raise RuntimeError(
                "Gold storage row-count verification "
                "failed: "
                f"expected={gold_count}, "
                f"stored={stored_gold_count}"
            )

        # ---------------------------------------------------------------
        # STEP 10: Validate stored Gold itself
        # ---------------------------------------------------------------

        GoldValidator.validate(stored_gold_df)

        # ---------------------------------------------------------------
        # STEP 11: Final independent reconciliation
        # ---------------------------------------------------------------

        final_reconciliation = SilverGoldReconciliation.reconcile(
            silver_df=silver_df,
            gold_df=stored_gold_df,
        )

        print("\nFinal reconciliation : PASS")

        # ---------------------------------------------------------------
        # STEP 12: Final report
        # ---------------------------------------------------------------

        print("\n" + "=" * 80)
        print("SILVER → GOLD PIPELINE " "COMPLETED SUCCESSFULLY")
        print("=" * 80)

        print(f"Silver input rows        : " f"{silver_count}")

        print(f"Eligible Silver rows     : " f"{final_reconciliation['eligible_silver_rows']}")

        print(f"Gold rows                : " f"{stored_gold_count}")

        print(f"Distinct vehicles        : " f"{final_reconciliation['distinct_silver_vehicles']}")

        print(f"Gold event count         : " f"{final_reconciliation['actual_event_count']}")

        print(f"Gold table               : " f"{GOLD_FULL_TABLE}")

        print(f"Gold path                : " f"{GOLD_PATH}")

        print("=" * 80)

        logger.info("Silver → Gold pipeline completed " "successfully")

    finally:

        try:
            silver_df.unpersist()
        except Exception:
            pass

        try:
            gold_df.unpersist()
        except Exception:
            pass

        try:
            stored_gold_df.unpersist()
        except Exception:
            pass

        spark.stop()


if __name__ == "__main__":
    main()
