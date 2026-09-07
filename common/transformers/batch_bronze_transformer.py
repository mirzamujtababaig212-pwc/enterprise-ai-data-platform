from pyspark.sql.functions import (
    current_timestamp,
    lit,
)

from common.transformers.base_transformer import (
    BaseTransformer,
)


class BatchBronzeTransformer(BaseTransformer):

    REQUIRED_COLUMNS = [
        "vehicle_id",
        "event_time",
        "latitude",
        "longitude",
        "speed",
        "rpm",
        "fuel_level",
        "battery",
        "engine_temperature",
        "gear",
    ]

    @staticmethod
    def transform(df):

        missing = [
            column for column in BatchBronzeTransformer.REQUIRED_COLUMNS if column not in df.columns
        ]

        if missing:
            raise RuntimeError("Batch input missing required columns: " + ", ".join(missing))

        result = (
            df.withColumn(
                "kafka_key",
                lit(None).cast("string"),
            )
            .withColumn(
                "kafka_topic",
                lit("batch"),
            )
            .withColumn(
                "kafka_partition",
                lit(None).cast("integer"),
            )
            .withColumn(
                "kafka_offset",
                lit(None).cast("long"),
            )
            .withColumn(
                "kafka_timestamp",
                lit(None).cast("timestamp"),
            )
            .withColumn(
                "raw_value",
                lit(None).cast("string"),
            )
            .withColumn(
                "ingestion_time",
                current_timestamp(),
            )
        )

        return result
