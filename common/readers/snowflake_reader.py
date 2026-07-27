class SnowflakeReader(BaseReader):
    @staticmethod
    def read_table(
        spark,
        table
    ):
        return (
            spark.read
            .format("snowflake")
            .option(
                "dbtable",
                table
            )
            .load()
        )
