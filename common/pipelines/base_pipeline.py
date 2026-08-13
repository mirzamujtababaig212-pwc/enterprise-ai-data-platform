import time
from abc import ABC
from typing import ClassVar

from common.logging.logger import get_logger
from common.pipelines.pipeline_config import PipelineConfig


logger = get_logger(__name__)


class BasePipeline(ABC):

    CONFIG: ClassVar[PipelineConfig]

    def __init__(
        self,
        spark,
        reader,
        validator,
        writer,
        transformer,
        metrics=None,
        dlq=None,
    ):
        self.spark = spark
        self.reader = reader
        self.validator = validator
        self.writer = writer
        self.transformer = transformer
        self.metrics = metrics
        self.dlq = dlq
        self.config = self.CONFIG

    # ================================================================
    # PUBLIC EXECUTION METHODS
    # ================================================================

    def run_batch(self):

        logger.info(
            "Starting BATCH execution for %s",
            self.config.pipeline_name,
        )

        try:
            self.initialize()

            source_df = self.read()

            batch_id = int(time.time() * 1000)

            self.process_batch(
                batch_df=source_df,
                batch_id=batch_id,
            )

        finally:
            self.cleanup()

    def run_stream(self):

        logger.info(
            "Starting STREAMING execution for %s",
            self.config.pipeline_name,
        )

        try:
            self.initialize()

            source_df = self.read()

            return self.write_stream(source_df)

        finally:
            self.cleanup()

    def run(self, mode="stream"):

        if mode == "batch":
            return self.run_batch()

        if mode == "stream":
            return self.run_stream()

        raise ValueError(f"Unsupported execution mode: {mode}. " "Use 'batch' or 'stream'.")

    # ================================================================
    # LIFECYCLE
    # ================================================================

    def initialize(self):

        logger.info(
            "Starting %s Pipeline",
            self.config.pipeline_name,
        )

    def cleanup(self):

        logger.info(
            "Cleaning up %s Pipeline",
            self.config.pipeline_name,
        )

    # ================================================================
    # READ
    # ================================================================

    def read(self):

        return self.reader.read(self.spark)

    # ================================================================
    # VALIDATION
    # ================================================================

    def validate(self, batch_df):

        if not self.config.enable_validation:
            return batch_df, None

        return self.validator.validate(batch_df)

    # ================================================================
    # INVALID RECORD HANDLING
    # ================================================================

    def handle_invalid_records(self, invalid_df):

        if invalid_df is not None and self.config.enable_dlq and self.dlq is not None:
            self.dlq.write(invalid_df)

    # ================================================================
    # METRICS
    # ================================================================

    def collect_metrics(
        self,
        pipeline,
        batch_id,
        batch_df,
        rejected_df=None,
        attempt=0,
        transform_duration=0,
        validation_duration=0,
        write_duration=0,
        dlq_duration=0,
        pipeline_duration=0,
    ):

        if not self.config.enable_metrics or self.metrics is None:
            return

        self.metrics.record_batch(
            pipeline=pipeline,
            batch_id=batch_id,
            batch_df=batch_df,
            rejected_df=rejected_df,
            retry_count=attempt,
            transform_duration=transform_duration,
            validation_duration=validation_duration,
            write_duration=write_duration,
            dlq_duration=dlq_duration,
            pipeline_duration=pipeline_duration,
        )

    # ================================================================
    # MICRO-BATCH / BATCH PROCESSING
    # ================================================================

    def process_batch(
        self,
        batch_df,
        batch_id,
    ):

        for attempt in range(self.config.retries):

            logger.info(
                "Processing %s Batch %s Attempt %s",
                self.config.pipeline_name,
                batch_id,
                attempt + 1,
            )

            pipeline_start = time.time()

            try:

                # ----------------------------------------------------
                # 1. TRANSFORM
                # ----------------------------------------------------

                transform_start = time.time()

                transformed_df = self.transformer.transform(batch_df)

                transform_duration = time.time() - transform_start

                # ----------------------------------------------------
                # 2. VALIDATE
                # ----------------------------------------------------

                validation_start = time.time()

                valid_df, invalid_df = self.validate(transformed_df)

                validation_duration = time.time() - validation_start

                # ----------------------------------------------------
                # 3. WRITE VALID RECORDS
                # ----------------------------------------------------

                write_start = time.time()

                if valid_df is not None:
                    self.writer.write(valid_df)

                write_duration = time.time() - write_start

                # ----------------------------------------------------
                # 4. DLQ
                # ----------------------------------------------------

                dlq_start = time.time()

                self.handle_invalid_records(invalid_df)

                dlq_duration = time.time() - dlq_start

                # ----------------------------------------------------
                # 5. METRICS
                # ----------------------------------------------------

                pipeline_duration = time.time() - pipeline_start

                self.collect_metrics(
                    pipeline=self.config.pipeline_name,
                    batch_id=batch_id,
                    batch_df=valid_df,
                    rejected_df=invalid_df,
                    attempt=attempt,
                    transform_duration=transform_duration,
                    validation_duration=validation_duration,
                    write_duration=write_duration,
                    dlq_duration=dlq_duration,
                    pipeline_duration=pipeline_duration,
                )

                logger.info(
                    "%s Batch %s completed successfully",
                    self.config.pipeline_name,
                    batch_id,
                )

                return

            except Exception:

                logger.exception(
                    "%s batch %s failed",
                    self.config.pipeline_name,
                    batch_id,
                )

                if attempt == self.config.retries - 1:
                    raise

                delay = self.config.retry_delay ** (attempt + 1)

                logger.info(
                    "Retrying in %s seconds",
                    delay,
                )

                time.sleep(delay)

    # ================================================================
    # STREAMING
    # ================================================================

    def write_stream(self, df):

        query = self.writer.write_stream(
            df=df,
            foreach_batch=self.process_batch,
            checkpoint=self.config.checkpoint,
            output_mode=self.config.output_mode,
            query_name=self.config.query_name,
            trigger=self.config.trigger,
        )

        return query.awaitTermination()
