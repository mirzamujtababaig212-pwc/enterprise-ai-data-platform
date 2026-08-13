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

    def transform(self, df):

        missing = [c for c in self.REQUIRED_COLUMNS if c not in df.columns]

        if missing:
            raise RuntimeError("Missing required columns: " + ", ".join(missing))

        result = (
            df.withColumn(
                "vehicle_id",
                trim(col("vehicle_id")),
            )
            .withColumn(
                "speed_category",
                when(
                    col("speed") < 20,
                    "LOW",
                )
                .when(
                    col("speed") < 60,
                    "NORMAL",
                )
                .otherwise("HIGH"),
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

        return result
