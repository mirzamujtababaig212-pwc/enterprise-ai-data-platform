from common.validation.base_validator import BaseValidator


class SchemaValidator(BaseValidator):
    def __init__(self, expected_columns):
        self.expected_columns = expected_columns

    def validate(self, df):
        missing_columns = [column for column in self.expected_columns if column not in df.columns]

        if missing_columns:
            raise RuntimeError("Missing required columns: " + ", ".join(missing_columns))

        return df, None
