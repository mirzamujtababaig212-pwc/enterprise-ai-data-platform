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
        "status",
        "event_timestamp",
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
                "topic",
                lit("batch"),
            )
            .withColumn(
                "partition",
                lit(None).cast("integer"),
            )
            .withColumn(
                "offset",
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
