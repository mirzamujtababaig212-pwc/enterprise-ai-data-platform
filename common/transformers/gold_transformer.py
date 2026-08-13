from pyspark.sql.functions import (
    avg,
    count,
    max,
)

from common.transformers.base_transformer import BaseTransformer


class GoldTransformer(BaseTransformer):

    def transform(self, df):
        try:
            return df.groupBy("vehicle_id").agg(
                avg("speed").alias("avg_speed"),
                max("speed").alias("max_speed"),
                avg("fuel_level").alias("avg_fuel_level"),
                avg("battery").alias("avg_battery"),
                max("engine_temperature").alias("max_engine_temperature"),
                count("*").alias("total_events"),
            )

        except Exception as exc:
            raise RuntimeError(f"Gold transformation failed: {exc}") from exc
