from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("Vehicle Batch Ingestion").getOrCreate()

spark.sparkContext.setLogLevel("WARN")

vehicle_df = (
    spark.read.option("header", True)
    .option("inferSchema", True)
    .csv("data/raw/vehicle_data.csv")
)

print("=" * 60)
print("Vehicle Dataset")
print("=" * 60)

vehicle_df.printSchema()
vehicle_df.show(10, truncate=False)

(vehicle_df.write.mode("overwrite").parquet("data/bronze"))

print("=" * 60)
print("Bronze Layer Created Successfully")
print("=" * 60)

spark.stop()
