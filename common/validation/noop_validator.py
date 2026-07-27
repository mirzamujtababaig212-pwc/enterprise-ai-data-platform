from common.validation.base_validator import BaseValidator


class NoOpValidator(BaseValidator):
    def validate(self, df):
        return df, None
