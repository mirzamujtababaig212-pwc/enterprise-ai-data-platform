import time

from common.logging.logger import get_logger
from common.validation.base_validator import BaseValidator
from common.validation.rules.duplicate_rule import DuplicateRule
from common.validation.rules.not_null_rule import NotNullRule
from common.validation.rules.regex_rule import RegexRule

logger = get_logger(__name__)


class CompositeValidator(BaseValidator):
    def __init__(self, validators=None):
        self.validators = validators or []

    def add_validator(self, validator):
        self.validators.append(validator)

    def validate(self, df):
        start = time.time()
        valid_df = df
        invalid_frames = []
        for validator in self.validators:
            print(f"\nRunning {type(validator).__name__}")
            print("Input:", valid_df.count())
            valid_df, invalid_df = validator.validate(valid_df)
            print("Valid:", valid_df.count())
            print("Invalid:", invalid_df.count() if invalid_df is not None else 0)
            if invalid_df is not None:
                invalid_frames.append(invalid_df)
            logger.info(
                "%s Valid=%s Invalid=%s",
                type(validator).__name__,
                valid_df.count(),
                invalid_df.count() if invalid_df is not None else 0,
            )
        if invalid_frames:
            final_invalid = invalid_frames[0]
            for frame in invalid_frames[1:]:
                final_invalid = final_invalid.unionByName(frame)
        else:
            final_invalid = df.sparkSession.createDataFrame([], df.schema)
        invalid_count = final_invalid.count()
        duration = time.time() - start
        logger.info("Validation Summary")
        logger.info("Processed=%s", df.count())
        logger.info("Valid=%s", valid_df.count())
        logger.info("Rejected=%s", invalid_count)
        logger.info("Validation completed in %.2f sec", duration)
        return valid_df, final_invalid
