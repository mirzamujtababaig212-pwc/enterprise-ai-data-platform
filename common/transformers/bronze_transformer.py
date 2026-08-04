from pyspark.sql.functions import col, current_timestamp, from_json

from common.transformers.base_transformer import BaseTransformer
from spark.schemas.bronze_schema import bronze_schema


class BronzeTransformer(BaseTransformer):

    def transform(self, df):
        bronze_df = (
            df.select(from_json(col("value").cast("string"), bronze_schema).alias("json"))
            .select("json.*")
            .withColumn("ingestion_time", current_timestamp())
        )
        return bronze_df
