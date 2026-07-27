from pyspark.sql.functions import col


class ValidationRules:
    @staticmethod
    def required_columns():
        return [
            "vehicle_id",
            "timestamp",
            "speed",
            "fuel_level"
        ]
    @staticmethod
    def valid_speed():
        return (
            (col("speed") >= 0) &
            (col("speed") <= 250)
        )
    @staticmethod
    def valid_fuel():
        return (
            (col("fuel_level") >= 0) &
            (col("fuel_level") <= 100)
        )
    @staticmethod
    def valid_engine_temperature():
        return (
            (col("engine_temp") >= -40) &
            (col("engine_temp") <= 200)
        )
