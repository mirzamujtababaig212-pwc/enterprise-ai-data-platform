from common.readers.base_reader import BaseReader


class DeltaReader(BaseReader):

    def __init__(self, path, schema=None):
        self.path = path
        self.schema = schema  # accepted for API compatibility

    def read(self, spark):
        return spark.read.format("delta").load(self.path)
