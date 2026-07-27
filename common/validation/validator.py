from pyspark.sql.functions import col

from common.logging.logger import get_logger
from common.validation.rules import ValidationRules

logger = get_logger(__name__)

class DataQualityValidator:
	def validate(self, df):
		logger.info("Running data quality validation.")
		required = ValidationRules.required_columns()
		valid_df = df
		for column in required:
			valid_df = valid_df.filter(
				col(column).isNotNull()
			)
		valid_df=valid_df.filter(
			ValidationRules.valid_speed()
		)
		valid_df=valid_df.filter(
			ValidationRules.valid_fuel()
		)
		valid_df=valid_df.filter(
			ValidationRules.valid_engine_temperate()
		)
		invalid_df=df.subtract(valid_df)
		logger.info("Validation Completed.")
		return valid_df, invalid_df
