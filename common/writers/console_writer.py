from common.writers.base_writer import BaseWriter


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
        return (
            df.writeStream.outputMode("append")
            .format("console")
            .option("truncate", False)
            .option("numRows", 20)
            .start()
        )

    def write(self, df):
        self.write_batch(df)
