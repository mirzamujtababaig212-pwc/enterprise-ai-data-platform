from pyspark.sql.functions import col, trim, upper

from common.transformers.base_transformer import BaseTransformer


class SilverTransformer(BaseTransformer):

    REQUIRED_COLUMNS = [
        "vehicle_id",
        "status",
        "event_timestamp",
    ]

    def transform(self, df):
        missing = [c for c in self.REQUIRED_COLUMNS if c not in df.columns]

        if missing:
            raise RuntimeError(f"Missing required columns: {', '.join(missing)}")

        return (
            df.withColumn("vehicle_id", trim(col("vehicle_id")))
            .withColumn("status", upper(col("status")))
            .dropDuplicates(["vehicle_id", "event_timestamp"])
        )
