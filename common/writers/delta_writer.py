from common.writers.base_writer import BaseWriter


class DeltaWriter(BaseWriter):

    def __init__(
        self,
        table,
        checkpoint,
        mode="append",
        output_mode="append",
        trigger=None
    ):
        self.table = table
        self.mode = mode
        self.checkpoint = checkpoint
        self.output_mode = output_mode
        self.trigger = trigger

    def write_stream(
        self,
        df,
        foreach_batch
    ):
        writer = (
            df.writeStream
              .foreachBatch(foreach_batch)
              .option(
                    "checkpointLocation",
                    self.checkpoint
              )
              .outputMode(self.output_mode)
        )
        if self.trigger:
            writer = writer.trigger(
                **self.trigger
            )
        return writer.start()

    def write(self, df):
        self.write_batch(df)

    def write_batch(self, df):
        (
            df.write
              .format("delta")
              .mode(self.mode)
              .saveAsTable(self.table)
        )
