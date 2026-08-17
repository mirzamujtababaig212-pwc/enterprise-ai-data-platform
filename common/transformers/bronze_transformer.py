from pyspark.sql.functions import (
    col,
    current_timestamp,
    from_json,
)

from common.transformers.base_transformer import (
    BaseTransformer,
)
from spark.schemas.bronze_schema import (
    bronze_schema,
)


class BronzeTransformer(BaseTransformer):

    @staticmethod
    def transform(df):

        parsed_df = df.select(
            col("key").cast("string").alias("kafka_key"),
            col("value").cast("string").alias("raw_value"),
            col("topic").alias("kafka_topic"),
            col("partition").alias("kafka_partition"),
            col("offset").alias("kafka_offset"),
            col("timestamp").alias("kafka_timestamp"),
        )

        bronze_df = (
            parsed_df.withColumn(
                "json",
                from_json(
                    col("raw_value"),
                    bronze_schema,
                ),
            )
            .select(
                "kafka_key",
                "kafka_topic",
                "kafka_partition",
                "kafka_offset",
                "kafka_timestamp",
                "raw_value",
                col("json.*"),
            )
            .withColumn(
                "ingestion_time",
                current_timestamp(),
            )
        )

        return bronze_df
