from common.config.settings import Settings


class PostgresReader:
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
