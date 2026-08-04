from common.readers.base_reader import BaseReader


class JDBCReader(BaseReader):
    def __init__(self, url, table, properties):
        self.url = url
        self.table = table
        self.properties = properties

    def read(self, spark):
        return spark.read.jdbc(url=self.url, table=self.table, properties=self.properties)
