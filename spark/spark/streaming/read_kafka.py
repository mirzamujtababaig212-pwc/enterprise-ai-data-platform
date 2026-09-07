from common.spark.spark_builder import SparkSessionBuilder

spark = SparkSessionBuilder.build("VehicleTelemetryStreaming")

spark.sparkContext.setLogLevel("WARN")

print(f"Spark Version : {spark.version}")

df = (
    spark.readStream.format("kafka")
    .option("kafka.bootstrap.servers", "localhost:9094")
    .option("subscribe", "vehicle-telemetry")
    .option("startingOffsets", "earliest")
    .load()
)

query = (
    df.selectExpr(
        "CAST(key AS STRING)",
        "CAST(value AS STRING)",
        "topic",
        "partition",
        "offset",
        "timestamp",
    )
    .writeStream.format("console")
    .outputMode("append")
    .option("truncate", False)
    .start()
)

query.awaitTermination()
