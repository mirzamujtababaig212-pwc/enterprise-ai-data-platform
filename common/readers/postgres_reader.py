from common.config.settings import Settings
from common.readers.base_reader import BaseReader


class PostgresReader(BaseReader):
    def __init__(self, table):
        self.table = table

    def read(self, spark):
        return self.read_table(spark, self.table)

    @staticmethod
    def read_table(spark, table):
        return (
            spark.read.format("jdbc")
            .option("url", Settings.postgres.URL)
            .option("dbtable", table)
            .option("user", Settings.postgres.USER)
            .option("password", Settings.postgres.PASSWORD)
            .load()
        )
