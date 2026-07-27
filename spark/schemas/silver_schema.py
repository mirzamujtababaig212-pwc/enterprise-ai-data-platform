from pyspark.sql.types import *

silver_schema = StructType([
    StructField("vehicle_id", StringType(), True),
    StructField("event_time", TimestampType(), True),
    StructField("latitude", DoubleType(), True),
    StructField("longitude", DoubleType(), True),
    StructField("speed", DoubleType(), True),
    StructField("rpm", IntegerType(), True),
    StructField("fuel_level", DoubleType(), True),
    StructField("battery", DoubleType(), True),
    StructField("engine_temperature", DoubleType(), True),
    StructField("gear", IntegerType(), True),
    StructField("topic", StringType(), True),
    StructField("partition", IntegerType(), True),
    StructField("offset", LongType(), True),
    StructField("timestamp", TimestampType(), True),
    StructField("ingestion_timestamp", TimestampType(), True),
    StructField("speed_category", StringType(), True),
    StructField("fuel_status", StringType(), True),
    StructField("battery_status", StringType(), True)
])
