from pyspark.sql.functions import col


class DataQualityValidator:
    def validate(self, df):
        valid = (
            df.filter(col("vehicle_id").isNotNull())
            .filter(col("speed").between(0, 250))
            .filter(col("fuel_level").between(0, 100))
            .filter(col("battery").between(0, 100))
            .filter(col("engine_temperature").between(0, 160))
        )
        invalid = df.subtract(valid)
        return valid, invalid
