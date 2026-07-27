from common.readers.base_reader import BaseReader


class CSVReader(BaseReader):
    def __init__(
        self,
        path,
        header=True,
        schema=None,
    ):
        self.path = path
        self.header = header
        self.schema = schema
    def read(self, spark):
        reader = (
            spark.read
                 .option(
                     "header",
                     self.header
                 )
        )
        if self.schema:
            reader = reader.schema(
                self.schema
            )
        return reader.csv(
            self.path
        )
