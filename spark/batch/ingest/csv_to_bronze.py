from common.config.settings import Settings
from common.spark.spark_builder import SparkSessionBuilder

spark = SparkSessionBuilder.build("Vehicle Batch Ingestion")

spark.sparkContext.setLogLevel("WARN")

vehicle_df = (
    spark.read.option("header", True)
    .option("inferSchema", True)
    .csv(Settings.storage.RAW_VEHICLE_DATA_PATH)
)

print("=" * 60)
print("Vehicle Dataset")
print("=" * 60)

vehicle_df.printSchema()
vehicle_df.show(10, truncate=False)

(vehicle_df.write.mode("overwrite").parquet(Settings.storage.BATCH_BRONZE_PATH))

print("=" * 60)
print("Bronze Layer Created Successfully")
print("=" * 60)

spark.stop()
