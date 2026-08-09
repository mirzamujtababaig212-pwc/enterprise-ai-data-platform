from pyspark.sql import Window
from pyspark.sql.functions import lit, row_number

from common.validation.base_validator import BaseValidator


class DuplicateValidator(BaseValidator):
    def __init__(self, keys):
        self.keys = keys

    def validate(self, df):

        window = Window.partitionBy(*self.keys).orderBy(lit(1))

        numbered = df.withColumn("_rn", row_number().over(window))

        valid = numbered.filter("_rn = 1").drop("_rn")

        invalid = numbered.filter("_rn > 1").drop("_rn")

        return valid, invalid
