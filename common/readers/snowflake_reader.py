from common.readers.base_reader import BaseReader


class SnowflakeReader(BaseReader):
    @staticmethod
    def read_table(spark, table):
        return spark.read.format("snowflake").option("dbtable", table).load()
