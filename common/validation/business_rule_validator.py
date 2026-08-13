from pyspark.sql.functions import col

from common.validation.base_validator import BaseValidator


class BusinessRuleValidator(BaseValidator):

    def validate(self, df):

        required_columns = [
            "speed",
            "fuel_level",
        ]

        missing = [column for column in required_columns if column not in df.columns]

        if missing:

            raise RuntimeError(
                "Business rule validation requires " f"columns: {', '.join(missing)}"
            )

        invalid_condition = (
            col("speed").isNull()
            | (col("speed") < 0)
            | col("fuel_level").isNull()
            | (col("fuel_level") < 0)
        )

        invalid = df.filter(invalid_condition)

        valid = df.filter(~invalid_condition)

        return valid, invalid
