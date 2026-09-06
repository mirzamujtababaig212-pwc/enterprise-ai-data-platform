from common.readers.base_reader import BaseReader


class SnowflakeReader(BaseReader):
    def __init__(self, table):
        self.table = table

    def read(self, spark):
        return self.read_table(spark, self.table)

    @staticmethod
    def read_table(spark, table):
        return spark.read.format("snowflake").option("dbtable", table).load()
