from common.readers.base_reader import BaseReader


class DatabricksReader(BaseReader):
    """
    Spark-native Databricks table reader.

    The reader intentionally remains independent from Databricks
    authentication and control-plane configuration. In a Databricks
    Spark environment, table resolution is delegated to Spark's catalog
    through spark.table().
    """

    def __init__(self, table):
        self.table = table

    def read(self, spark):
        return self.read_table(
            spark=spark,
            table=self.table,
        )

    @staticmethod
    def read_table(
        spark,
        table,
    ):
        return spark.table(table)
