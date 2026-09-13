from __future__ import annotations

import time

from common.logging.logger import get_logger
from common.pipelines.base_pipeline import BasePipeline
from common.validation.data_quality import DataQualityValidator
from spark.schemas.gold_schema import GOLD_REQUIRED_COLUMNS
from spark.validation.gold_validator import GoldValidator
from spark.validation.silver_gold_reconciliation import SilverGoldReconciliation

logger = get_logger(__name__)


GOLD_DATABASE = "gold"
GOLD_TABLE = "vehicle_metrics"
GOLD_FULL_TABLE = f"{GOLD_DATABASE}.{GOLD_TABLE}"


class GoldPipeline(BasePipeline):
    """
    Domain-specific Silver -> Gold pipeline.

    The generic BasePipeline remains responsible for the common pipeline
    framework, while GoldPipeline owns Gold-specific data quality,
    semantic validation, reconciliation, and persisted-output verification.
    """

    def run_batch(self):
        logger.info(
            "Starting BATCH execution for %s",
            self.config.pipeline_name,
        )

        silver_df = None
        gold_df = None
        stored_gold_df = None

        try:
            self.initialize()

            # -----------------------------------------------------------
            # 1. READ SILVER
            # -----------------------------------------------------------

            silver_df = self.read().cache()
            silver_count = silver_df.count()

            logger.info(
                "Silver input rows for %s: %s",
                self.config.pipeline_name,
                silver_count,
            )

            if silver_count == 0:
                raise RuntimeError("Silver dataset is empty. Gold pipeline will not run.")

            # -----------------------------------------------------------
            # 2. TRANSFORM SILVER -> GOLD
            # -----------------------------------------------------------

            transform_start = time.time()

            gold_df = self.transformer.transform(silver_df).cache()

            transform_duration = time.time() - transform_start
            gold_count = gold_df.count()

            logger.info(
                "Gold transformation produced %s rows",
                gold_count,
            )

            if gold_count == 0:
                raise RuntimeError("Gold transformation produced zero rows.")

            # -----------------------------------------------------------
            # 3. GENERIC GOLD DATA QUALITY
            # -----------------------------------------------------------

            validation_start = time.time()

            validator = DataQualityValidator(
                dataframe=gold_df,
                dataset_name=GOLD_FULL_TABLE,
            )

            validator.check_row_count(minimum=1)

            for column in (
                "vehicle_id",
                "event_count",
                "avg_speed",
                "min_speed",
                "max_speed",
                "first_event_time",
                "last_event_time",
            ):
                validator.check_not_null(column)

            validator.check_min_value("event_count", 1)
            validator.check_min_value("avg_speed", 0.0)
            validator.check_min_value("min_speed", 0.0)
            validator.check_min_value("max_speed", 0.0)

            validator.check_unique(["vehicle_id"])
            validator.check_columns(GOLD_REQUIRED_COLUMNS)
            validator.validate_or_raise()

            # -----------------------------------------------------------
            # 4. GOLD SEMANTIC VALIDATION
            # -----------------------------------------------------------

            GoldValidator.validate_schema(gold_df)
            GoldValidator.validate_not_null(gold_df)
            GoldValidator.validate_numeric_ranges(gold_df)
            GoldValidator.validate_speed_order(gold_df)
            GoldValidator.validate_time_order(gold_df)
            GoldValidator.validate_unique_vehicle(gold_df)

            validation_duration = time.time() - validation_start

            # -----------------------------------------------------------
            # 5. INDEPENDENT SILVER -> GOLD RECONCILIATION
            # -----------------------------------------------------------

            reconciliation = SilverGoldReconciliation.reconcile(
                silver_df=silver_df,
                gold_df=gold_df,
            )

            logger.info(
                "Silver -> Gold reconciliation: %s",
                reconciliation,
            )

            # -----------------------------------------------------------
            # 6. WRITE GOLD
            # -----------------------------------------------------------

            write_start = time.time()

            self.writer.write(gold_df)

            write_duration = time.time() - write_start

            # -----------------------------------------------------------
            # 7. READ PERSISTED GOLD
            # -----------------------------------------------------------

            gold_table = self.writer.table

            stored_gold_df = self.spark.table(gold_table).cache()
            stored_gold_count = stored_gold_df.count()

            if stored_gold_count != gold_count:
                raise RuntimeError(
                    "Gold storage row-count verification failed: "
                    f"expected={gold_count}, "
                    f"stored={stored_gold_count}"
                )

            # -----------------------------------------------------------
            # 8. VALIDATE PERSISTED GOLD
            # -----------------------------------------------------------

            GoldValidator.validate(stored_gold_df)

            # -----------------------------------------------------------
            # 9. FINAL INDEPENDENT RECONCILIATION
            # -----------------------------------------------------------

            final_reconciliation = SilverGoldReconciliation.reconcile(
                silver_df=silver_df,
                gold_df=stored_gold_df,
            )

            logger.info(
                "Final Silver -> Gold reconciliation: %s",
                final_reconciliation,
            )

            # -----------------------------------------------------------
            # 10. METRICS
            # -----------------------------------------------------------

            batch_id = int(time.time() * 1000)
            pipeline_duration = transform_duration + validation_duration + write_duration

            self.collect_metrics(
                pipeline=self.config.pipeline_name,
                batch_id=batch_id,
                batch_df=gold_df,
                rejected_df=None,
                attempt=0,
                transform_duration=transform_duration,
                validation_duration=validation_duration,
                write_duration=write_duration,
                pipeline_duration=pipeline_duration,
            )

            logger.info(
                "Gold pipeline completed successfully: "
                "silver_rows=%s eligible_silver_rows=%s gold_rows=%s",
                silver_count,
                final_reconciliation["eligible_silver_rows"],
                stored_gold_count,
            )

            return final_reconciliation

        finally:
            if stored_gold_df is not None:
                stored_gold_df.unpersist()

            if gold_df is not None:
                gold_df.unpersist()

            if silver_df is not None:
                silver_df.unpersist()

            self.cleanup()
