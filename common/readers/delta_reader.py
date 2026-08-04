from pyspark.errors import AnalysisException

from common.readers.base_reader import BaseReader


class DeltaReader(BaseReader):

    def __init__(self, path, schema=None):
        self.path = path
        self.schema = schema  # kept only for API compatibility

    def read(self, spark):
        try:
            # Delta always reads its schema from the transaction log.
            return spark.read.format("delta").load(self.path)

        except AnalysisException as e:
            raise RuntimeError(str(e)) from e
