from common.readers.base_reader import BaseReader


class SnowflakeReader(BaseReader):
    """
    Spark Snowflake reader.

    Connection options are supplied by the ReaderBuilder so the reader
    remains independent from the configuration source.
    """

    def __init__(
        self,
        options,
        table,
    ):
        self.options = options
        self.table = table

    def read(self, spark):
        return self.read_table(
            spark=spark,
            options=self.options,
            table=self.table,
        )

    @staticmethod
    def read_table(
        spark,
        options,
        table,
    ):
        return spark.read.format("snowflake").options(**options).option("dbtable", table).load()
