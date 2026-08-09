from pyspark.sql.functions import avg, count, max, min

from common.transformers.base_transformer import BaseTransformer


class GoldTransformer(BaseTransformer):
    def transform(self, df):
        try:
            return df.groupBy("vehicle_id").agg(
                avg("speed").alias("avg_speed"),
                max("speed").alias("max_speed"),
                min("speed").alias("min_speed"),
                count("*").alias("events"),
            )
        except Exception as e:
            raise RuntimeError(str(e)) from e
