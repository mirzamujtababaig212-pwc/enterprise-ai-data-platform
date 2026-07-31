import time

from common.logging.logger import get_logger
from common.writers.base_writer import BaseWriter

logger = get_logger(__name__)


class DeltaWriter(BaseWriter):

    def __init__(
        self, table, checkpoint, mode="append", output_mode="append", trigger=None
    ):
        self.table = table
        self.mode = mode
        self.checkpoint = checkpoint
        self.output_mode = output_mode
        self.trigger = trigger

    def write_stream(self, df, foreach_batch):
        writer = (
            df.writeStream.foreachBatch(foreach_batch)
            .option("checkpointLocation", self.checkpoint)
            .outputMode(self.output_mode)
        )
        if self.trigger:
            writer = writer.trigger(**self.trigger)
        return writer.start()

    def write(self, df):
        self.write_batch(df)

    def write_batch(self, df):
        start = time.time()
        (df.write.format("delta").mode(self.mode).saveAsTable(self.table))
        duration = time.time() - start
        rows = df.count()
        logger.info("Rows Written=%s", df.count())

        logger.info("Write Duration=%.2f", duration)
