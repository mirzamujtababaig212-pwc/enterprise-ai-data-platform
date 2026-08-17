import os

from common.spark.spark_builder import SparkSessionBuilder

spark = SparkSessionBuilder.build("VerifyDelta")

print("\n===== SPARK VERSION =====")
print(spark.version)

print("\n===== DELTA TEST =====")

spark.sql("CREATE DATABASE IF NOT EXISTS silver")

spark.sql(
    """
    CREATE TABLE IF NOT EXISTS silver.vehicle_events
    USING DELTA
    LOCATION 'data/silver_delta'
"""
)

print("\n===== TABLES =====")
spark.sql("SHOW TABLES IN silver").show(truncate=False)

print("\n===== TABLE DESCRIPTION =====")
spark.sql("DESCRIBE EXTENDED silver.vehicle_events").show(
    truncate=False,
    n=100,
)

print("\n===== DELTA DIRECTORY =====")

delta_log = "data/silver_delta/_delta_log"

print("Delta log exists:", os.path.isdir(delta_log))

if os.path.isdir(delta_log):
    print("Delta log files:")
    for name in sorted(os.listdir(delta_log)):
        print("  ", name)

print("\n===== FINISHED =====")

spark.stop()
