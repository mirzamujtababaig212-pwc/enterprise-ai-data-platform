from common.writers.base_writer import BaseWriter
from common.writers.streaming import apply_stream_options


class IcebergWriter(BaseWriter):
    def __init__(self, table):
        self.table = table

    def write_batch(self, df):
        (df.writeTo(self.table).append())

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
