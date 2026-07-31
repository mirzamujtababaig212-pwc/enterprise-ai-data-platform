from pyspark.sql.functions import col

from common.validation.base_validator import BaseValidator


class NullValidator(BaseValidator):
    def __init__(self, required_columns):
        self.required_columns = required_columns

    def validate(self, df):
        condition = None
        for c in self.required_columns:
            expr = col(c).isNull() | (col(c) == "")
            condition = expr if condition is None else (condition | expr)
        if condition is None:
            return df, df.limit(0)
        invalid = df.filter(condition)
        valid = df.filter(~condition)
        return valid, invalid
