from pyspark.sql.utils import AnalysisException

from common.readers.base_reader import BaseReader


class DeltaReader(BaseReader):

    def __init__(
        self,
        path: str,
        schema=None,
        table=None,
    ):
        self.path = path
        self.schema = schema
        self.table = table

    def read(self, spark):

        try:
            return spark.read.format("delta").load(self.path)

        except AnalysisException as exc:
            raise RuntimeError(f"Failed to read Delta path: {self.path}") from exc

    def read_stream(self, spark):

        try:
            return spark.readStream.format("delta").load(self.path)

        except AnalysisException as exc:
            raise RuntimeError(f"Failed to read Delta stream path: {self.path}") from exc
