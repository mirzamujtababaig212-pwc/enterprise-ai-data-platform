from pyspark.sql.types import (DoubleType, IntegerType, LongType, StringType,
                               StructField, StructType, TimestampType)

gold_schema = StructType(
    [
        StructField("vehicle_id", StringType(), True),
        StructField("avg_speed", DoubleType(), True),
        StructField("max_speed", DoubleType(), True),
        StructField("avg_fuel_level", DoubleType(), True),
        StructField("avg_battery", DoubleType(), True),
        StructField("max_engine_temperature", DoubleType(), True),
        StructField("total_events", LongType(), True),
    ]
)
