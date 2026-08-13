from common.logging.logger import get_logger
from common.readers.base_reader import BaseReader


logger = get_logger(__name__)


class ParquetReader(BaseReader):

    def __init__(
        self,
        path,
        schema=None,
    ):
        self.path = path
        self.schema = schema

    def read(self, spark):

        try:

            reader = spark.read

            if self.schema:
                reader = reader.schema(self.schema)

            return reader.parquet(self.path)

        except Exception as exc:

            raise RuntimeError(f"Failed to read Parquet " f"from {self.path}: {exc}") from exc
