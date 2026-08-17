from common.spark.spark_builder import SparkSessionBuilder

spark = SparkSessionBuilder.build("TestDeltaTable")

spark.sql("CREATE DATABASE IF NOT EXISTS silver")

spark.sql(
    """
    CREATE TABLE IF NOT EXISTS silver.vehicle_events_test
    USING DELTA
    LOCATION 'data/silver_delta_test'
    """
)

print("\n===== TABLES =====")

spark.sql("SHOW TABLES IN silver").show(truncate=False)

print("\n===== TABLE DATA =====")

spark.sql("SELECT * FROM silver.vehicle_events_test LIMIT 10").show(truncate=False)

print("\n===== TABLE DESCRIPTION =====")

spark.sql("DESCRIBE EXTENDED silver.vehicle_events_test").show(
    truncate=False,
    n=100,
)

spark.stop()
