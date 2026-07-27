from pyspark.sql.functions import col, trim, upper

from common.transformers.base_transformer import BaseTransformer


class SilverTransformer(BaseTransformer):
    
    def transform(self,df):
        silver_df = (
            df
            .withColumn(
                "vehicle_id",
                trim(col("vehicle_id"))
            )
            .withColumn(
                "status",
                upper(col("status"))
            )
            .dropDuplicates(
                [
                    "vehicle_id",
                    "event_timestamp"
                ]
            )
        )
        return silver_df
