from common.writers.base_writer import BaseWriter


class IcebergWriter(BaseWriter):
    def __init__(self, table):
        self.table = table

    def write_batch(
        self,
        df
    ):
        (
            df.writeTo(self.table)
              .append()
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
