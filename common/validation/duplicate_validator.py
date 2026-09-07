from pyspark.sql import Window
from pyspark.sql.functions import col, lit, row_number

from common.validation.base_validator import BaseValidator


class DuplicateValidator(BaseValidator):

    def __init__(self, keys, order_columns=None):
        self.keys = keys
        self.order_columns = order_columns or []

    def validate(self, df):

        if self.order_columns:
            ordering = [col(column).asc_nulls_last() for column in self.order_columns]

            window = Window.partitionBy(*self.keys).orderBy(*ordering)

        else:
            window = Window.partitionBy(*self.keys).orderBy(lit(1))

        numbered = df.withColumn(
            "_rn",
            row_number().over(window),
        )

        valid = numbered.filter(col("_rn") == 1).drop("_rn")

        invalid = numbered.filter(col("_rn") > 1).drop("_rn")

        return valid, invalid
