import time

from common.logging.logger import get_logger
from common.writers.base_writer import BaseWriter
from common.writers.streaming import apply_stream_options

logger = get_logger(__name__)


class SnowflakeWriter(BaseWriter):
    def __init__(
        self,
        options,
        table,
        mode="append",
    ):
        self.options = options
        self.table = table
        self.mode = mode

    def write_batch(self, df):
        start = time.time()
        (
            df.write.format("snowflake")
            .options(**self.options)
            .option("dbtable", self.table)
            .mode(self.mode)
            .save()
        )
        duration = time.time() - start
        logger.info("Rows Written=%s", df.count())
        logger.info("Write Duration=%.2f sec", duration)

    def write_stream(
        self,
        df,
        foreach_batch,
        checkpoint=None,
        output_mode=None,
        query_name=None,
        trigger=None,
    ):
        writer = df.writeStream.foreachBatch(foreach_batch)

        writer = apply_stream_options(
            writer=writer,
            checkpoint=checkpoint,
            output_mode=output_mode,
            query_name=query_name,
            trigger=trigger,
        )

        return writer.start()

    def write(self, df):
        self.write_batch(df)
