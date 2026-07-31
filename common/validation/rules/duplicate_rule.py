from pyspark.sql.functions import lit, row_number
from pyspark.sql.window import Window

from common.validation.base_validator import BaseValidator


class DuplicateRule(BaseValidator):

    def __init__(self, columns):
        self.columns = columns

    def validate(self, df):

        window = Window.partitionBy(*self.columns).orderBy(lit(1))

        numbered = df.withColumn("__row_number", row_number().over(window))

        valid_df = numbered.filter("__row_number = 1").drop("__row_number")

        invalid_df = numbered.filter("__row_number > 1").drop("__row_number")

        return valid_df, invalid_df
