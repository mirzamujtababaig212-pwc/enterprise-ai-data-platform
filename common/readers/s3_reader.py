from common.readers.base_reader import BaseReader


class S3Reader(BaseReader):
    """
    Spark-native reader for Parquet datasets stored in Amazon S3.

    S3 authentication and filesystem access are delegated to Spark/Hadoop
    configuration. Object-level boto3 operations remain the responsibility
    of S3Storage.
    """

    def __init__(
        self,
        path,
        schema=None,
    ):
        self.path = path
        self.schema = schema

    def read(self, spark):
        return self.read_table(
            spark=spark,
            path=self.path,
            schema=self.schema,
        )

    @staticmethod
    def read_table(
        spark,
        path,
        schema=None,
    ):
        reader = spark.read

        if schema:
            reader = reader.schema(schema)

        return reader.parquet(path)
