from common.spark.spark_builder import SparkSessionBuilder


spark = SparkSessionBuilder.build("TestDeltaWrite")

source_path = "data/bronze"
delta_path = "data/silver_delta_test"

print("\n===== READING BRONZE SAMPLE =====")

df = spark.read.format("parquet").load(source_path)

df.printSchema()

print("\n===== SAMPLE DATA =====")

df.limit(5).show(truncate=False)

print("\n===== WRITING DELTA =====")

(df.limit(100).write.format("delta").mode("overwrite").save(delta_path))

print("\n===== DELTA WRITE COMPLETE =====")

print("\n===== READING DELTA =====")

(spark.read.format("delta").load(delta_path).show(5, truncate=False))

spark.stop()
