import time

from common.logging.logger import get_logger
from common.writers.base_writer import BaseWriter

logger = get_logger(__name__)


class FabricWriter(BaseWriter):
    def __init__(
        self,
        table,
        checkpoint=None,
        mode="append",
        output_mode="append",
    ):
        self.table = table
        self.mode = mode
        self.checkpoint = checkpoint
        self.output_mode = output_mode

    def write_batch(self, df):
        start = time.time()
        (df.write.format("delta").mode(self.mode).saveAsTable(self.table))
        duration = time.time() - start
        logger.info("Rows Written=%s", df.count())
        logger.info("Write Duration=%.2f sec", duration)

    def write_stream(self, df, foreach_batch):
        writer = df.writeStream.foreachBatch(
            lambda batch, _: self.write_batch(batch)
        ).start()
        if self.checkpoint:
            writer = writer.option(
                "checkpointLocation",
                self.checkpoint,
            )
        return writer.start()

    def write(self, df):
        self.write_batch(df)
