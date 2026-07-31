from common.validation.base_validator import BaseValidator
from common.validation.rules.duplicate_rule import DuplicateRule
from common.validation.rules.not_null_rule import NotNullRule
from common.validation.rules.regex_rule import RegexRule


class CompositeValidator:
    def __init__(self):
        self.rules: list[BaseValidator] = [
            NotNullRule(columns=["vehicle_id", "timestamp", "speed", "fuel_level"]),
            RegexRule(
                column="vehicle_id",
                pattern=r"^[A-Za-z0-9]+$",
            ),
            DuplicateRule(
                columns=[
                    "vehicle_id",
                    "timestamp",
                ]
            ),
        ]

    def validate(self, df):
        valid = df
        invalid = None
        for rule in self.rules:
            valid, rejected = rule.validate(valid)
            if rejected is not None:
                invalid = rejected if invalid is None else invalid.union(rejected)
        return valid, invalid
