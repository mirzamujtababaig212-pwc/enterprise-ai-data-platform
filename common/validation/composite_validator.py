from common.validation.base_validator import BaseValidator


class CompositeValidator(BaseValidator):
    def __init__(self, validators=None):
        self.validators = validators or []

    def add_validator(self, validator):
        self.validators.append(validator)

    def validate(self, df):
        valid_df = df
        invalid_frames = []
        for validator in self.validators:
            print(f"\nRunning {type(validator).__name__}")
            print("Input:", valid_df.count())
            valid_df, invalid_df = validator.validate(valid_df)
            print("Valid:", valid_df.count())
            print(
                "Invalid:",
                invalid_df.count() if invalid_df is not None else None
            )
            if invalid_df is not None:
                invalid_frames.append(invalid_df)
        if invalid_frames:
            final_invalid = invalid_frames[0]
            for frame in invalid_frames[1:]:
                final_invalid = final_invalid.unionByName(frame)
        else:
            final_invalid = (
                df.sparkSession
                  .createDataFrame([], df.schema)
            )
        return valid_df, final_invalid
