from pyspark.sql import DataFrame
from pyspark.sql import functions as F


class BronzeToSilverTransformer:
    """
    Transforms Bronze vehicle-event data into a
    cleaned and standardized Silver representation.
    """

    @staticmethod
    def transform(df: DataFrame) -> DataFrame:
        transformed = df.select(
            F.col("vehicle_id").cast("string").alias("vehicle_id"),
            F.to_timestamp(F.col("event_time")).alias("event_time"),
            F.col("speed").cast("double").alias("speed"),
        )
        transformed = (
            transformed.filter(F.col("vehicle_id").isNotNull())
            .filter(F.col("event_time").isNotNull())
            .filter(F.col("speed").isNotNull())
            .filter(F.col("speed") >= 0)
        )
        transformed = transformed.dropDuplicates(
            [
                "vehicle_id",
                "event_time",
            ]
        )
        return transformed
