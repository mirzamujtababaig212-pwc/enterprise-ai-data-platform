import time

from common.logging.logger import get_logger
from common.validation.base_validator import BaseValidator


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

            validator_name = type(validator).__name__

            logger.info(
                "Running validator=%s",
                validator_name,
            )

            valid_df, invalid_df = validator.validate(valid_df)

            if invalid_df is not None:
                invalid_frames.append(invalid_df)

        if invalid_frames:

            final_invalid = invalid_frames[0]

            for frame in invalid_frames[1:]:

                final_invalid = final_invalid.unionByName(
                    frame,
                    allowMissingColumns=True,
                )

        else:

            final_invalid = df.sparkSession.createDataFrame(
                [],
                df.schema,
            )

        logger.info(
            "Validation completed in %.2f sec",
            time.time() - start,
        )

        return (
            valid_df,
            final_invalid,
        )
