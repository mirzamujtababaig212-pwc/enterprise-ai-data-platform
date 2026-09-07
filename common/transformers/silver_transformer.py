from pyspark.sql.functions import (
    col,
    trim,
    when,
)

from common.transformers.base_transformer import (
    BaseTransformer,
)


class SilverTransformer(BaseTransformer):

    REQUIRED_COLUMNS = [
        "vehicle_id",
        "event_time",
        "speed",
        "fuel_level",
        "battery",
        "engine_temperature",
    ]

    OPTIONAL_COLUMNS = [
        "kafka_key",
        "kafka_topic",
        "kafka_partition",
        "kafka_offset",
        "kafka_timestamp",
        "raw_value",
        "ingestion_time",
        "latitude",
        "longitude",
        "rpm",
        "gear",
    ]

    @staticmethod
    def transform(df):

        missing_required = [
            column for column in SilverTransformer.REQUIRED_COLUMNS if column not in df.columns
        ]

        if missing_required:
            raise ValueError("Missing required columns: " + ", ".join(missing_required))

        result_df = (
            df.withColumn(
                "vehicle_id",
                trim(col("vehicle_id")),
            )
            .withColumn(
                "speed_category",
                when(col("speed") < 20, "LOW").when(col("speed") < 60, "NORMAL").otherwise("HIGH"),
            )
            .withColumn(
                "fuel_status",
                when(
                    col("fuel_level") < 15,
                    "CRITICAL",
                )
                .when(
                    col("fuel_level") < 30,
                    "LOW",
                )
                .otherwise("NORMAL"),
            )
            .withColumn(
                "battery_status",
                when(
                    col("battery") < 20,
                    "CRITICAL",
                )
                .when(
                    col("battery") < 40,
                    "LOW",
                )
                .otherwise("NORMAL"),
            )
            .withColumn(
                "vehicle_status",
                when(
                    col("battery") < 20,
                    "BATTERY_CRITICAL",
                )
                .when(
                    col("fuel_level") < 15,
                    "FUEL_CRITICAL",
                )
                .when(
                    col("engine_temperature") > 110,
                    "ENGINE_OVERHEAT",
                )
                .otherwise("NORMAL"),
            )
            .dropDuplicates(
                [
                    "vehicle_id",
                    "event_time",
                ]
            )
        )

        output_columns = [
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
            "kafka_key",
            "kafka_topic",
            "kafka_partition",
            "kafka_offset",
            "kafka_timestamp",
            "raw_value",
            "ingestion_time",
            "speed_category",
            "fuel_status",
            "battery_status",
            "vehicle_status",
        ]

        existing_columns = [column for column in output_columns if column in result_df.columns]

        return result_df.select(*existing_columns)
