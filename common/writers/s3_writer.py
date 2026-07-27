from common.writers.base_writer import BaseWriter


class S3Writer(BaseWriter):
    def __init__(
        self,
        path
    ):
        self.path = path

    def write_batch(
        self,
        df
    ):
        (
            df.write
              .mode("append")
              .parquet(self.path)
        )

    def write_stream(
        self,
        df,
        foreach_batch
    ):
        return (
            df.writeStream
              .foreachBatch(foreach_batch)
              .start()
        )

    def write(self, df):
        self.write_batch(df)
