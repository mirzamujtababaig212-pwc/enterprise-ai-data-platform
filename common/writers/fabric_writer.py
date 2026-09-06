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

    def write_stream(
        self,
        df,
        foreach_batch,
        checkpoint=None,
        output_mode=None,
        query_name=None,
        trigger=None,
    ):
        effective_checkpoint = checkpoint or self.checkpoint
        effective_output_mode = output_mode or self.output_mode or "append"

        writer = df.writeStream.foreachBatch(foreach_batch).outputMode(effective_output_mode)

        if effective_checkpoint:
            writer = writer.option(
                "checkpointLocation",
                effective_checkpoint,
            )

        if query_name:
            writer = writer.queryName(query_name)

        if trigger:
            if trigger.get("processingTime"):
                writer = writer.trigger(processingTime=trigger["processingTime"])
            elif trigger.get("availableNow"):
                writer = writer.trigger(availableNow=True)
            elif trigger.get("once"):
                writer = writer.trigger(once=True)

        return writer.start()

    def write(self, df):
        self.write_batch(df)
