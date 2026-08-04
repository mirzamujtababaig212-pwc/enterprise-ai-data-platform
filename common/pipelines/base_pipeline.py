import time
from abc import ABC
from typing import ClassVar

from common.logging.logger import get_logger
from common.pipelines.pipeline_config import PipelineConfig

logger = get_logger(__name__)


class BasePipeline(ABC):
    CONFIG: ClassVar[PipelineConfig]

    def __init__(self, spark, reader, validator, writer, transformer, metrics, dlq):
        self.spark = spark
        self.reader = reader
        self.validator = validator
        self.writer = writer
        self.transformer = transformer
        self.metrics = metrics
        self.dlq = dlq
        self.config = self.CONFIG

    def run(self):
        """
        Standard pipeline lifecycle.
        """
        try:
            self.initialize()
            source_df = self.read()
            transformed_df = self.transformer.transform(source_df)
            self.write_stream(transformed_df)
        finally:
            self.cleanup()

    def initialize(self):
        """Optional initialization."""
        logger.info("Starting %s Pipeline", self.config.pipeline_name)

    def cleanup(self):  # noqa: B027
        """Cleanup resources."""
        raise NotImplementedError

    def validate(self, batch_df):
        if not self.config.enable_validation:
            return batch_df, None
        return self.validator.validate(batch_df)

    def write(self, df):
        self.writer.write_batch(df)

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
        if not self.config.enable_metrics:
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

    def handle_invalid_records(self, invalid_df):
        if invalid_df is not None and self.config.enable_dlq:
            self.dlq.write(invalid_df)

    def read(self):
        return self.reader.read(self.spark)

    def process_batch(self, batch_df, batch_id):
        for attempt in range(self.config.retries):
            logger.info("Processing %s Batch %s", self.config.pipeline_name, batch_id)
            try:
                pipeline_start = time.time()
                transform_start = time.time()
                transformed = self.transformer.transform(batch_df)
                transform_duration = time.time() - transform_start
                validation_start = time.time()
                valid_df, invalid_df = self.validator.validate(transformed)
                validation_duration = time.time() - validation_start
                output_df = self.transformer.transform(valid_df)
                write_start = time.time()
                self.writer.write(output_df)
                write_duration = time.time() - write_start
                dlq_start = time.time()
                self.handle_invalid_records(invalid_df)
                dlq_duration = time.time() - dlq_start
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

                break

            except Exception:
                logger.exception("%s batch %s failed", self.config.pipeline_name, batch_id)
                if attempt == self.config.retries - 1:
                    raise
                time.sleep(self.config.retry_delay ** (attempt + 1))
            pipeline_duration = round(time.time() - pipeline_start, 2)
            logger.info("Pipeline Duration = %.2f sec", pipeline_duration)

    def write_stream(self, df):
        return self.writer.write_stream(df=df, foreach_batch=self.process_batch).awaitTermination()
