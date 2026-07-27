class FabricReader(BaseReader):
    @staticmethod
    def read_table(
        spark,
        table
    ):
        return (
            spark.read
            .format("delta")
            .load(table)
        )
