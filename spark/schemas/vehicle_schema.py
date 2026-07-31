from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

vehicle_schema = StructType(
    [
        StructField("vehicle_id", StringType()),
        StructField("event_time", TimestampType()),
        StructField("latitude", DoubleType()),
        StructField("longitude", DoubleType()),
        StructField("speed", DoubleType()),
        StructField("rpm", IntegerType()),
        StructField("fuel_level", DoubleType()),
        StructField("battery", DoubleType()),
        StructField("engine_temperature", DoubleType()),
        StructField("gear", IntegerType()),
    ]
)
