from common.validation.base_validator import BaseValidator


class SchemaValidator(BaseValidator):
    def __init__(self, expected_columns):
        self.expected_columns = expected_columns

    def validate(self, df):
        if list(df.columns) != list(self.expected_columns):
            raise RuntimeError(f"Expected schema {self.expected_columns}, got {df.columns}")
        return df, None
