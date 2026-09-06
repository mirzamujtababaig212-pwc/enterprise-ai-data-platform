from common.writers.base_writer import BaseWriter
from common.writers.streaming import apply_stream_options


class ConsoleWriter(BaseWriter):
    def write_batch(self, df):
        df.show(truncate=False)

    def write_stream(
        self,
        df,
        foreach_batch,
        checkpoint=None,
        output_mode=None,
        query_name=None,
        trigger=None,
    ):
        writer = df.writeStream.format("console").option("truncate", False).option("numRows", 20)

        writer = apply_stream_options(
            writer=writer,
            checkpoint=checkpoint,
            output_mode=output_mode or "append",
            query_name=query_name,
            trigger=trigger,
        )

        return writer.start()

    def write(self, df):
        self.write_batch(df)
