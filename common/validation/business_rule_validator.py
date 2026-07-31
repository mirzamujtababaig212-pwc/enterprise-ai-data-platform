from pyspark.sql.functions import col

from common.validation.base_validator import BaseValidator


class BusinessRuleValidator(BaseValidator):

    def validate(self, df):
        invalid = df.filter((col("speed") < 0) | (col("fuel_level") < 0))
        valid = df.subtract(invalid)
        return valid, invalid
