from pyspark.sql.functions import col
from pyspark.sql.types import StringType

from common.validation.base_validator import BaseValidator


class NullValidator(BaseValidator):
    def __init__(self, columns):
        self.columns = columns

    def validate(self, df):
        invalid_condition = None
        for column in self.columns:
            field = df.schema[column]
            if isinstance(field.dataType, StringType):
                condition = col(column).isNull() | (col(column) == "")
            else:
                condition = col(column).isNull()

            if invalid_condition is None:
                invalid_condition = condition
            else:
                invalid_condition = invalid_condition | condition

        if invalid_condition is None:
            return df, df.limit(0)
        invalid = df.filter(invalid_condition)
        valid = df.filter(~invalid_condition)
        return valid, invalid
