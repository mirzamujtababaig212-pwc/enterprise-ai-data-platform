from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("ReadBronze").getOrCreate()

df = spark.read.parquet("data/bronze")

df.printSchema()

df.show(truncate=False)
