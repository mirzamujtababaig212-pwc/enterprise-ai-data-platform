from delta import configure_spark_with_delta_pip
from pyspark.sql import SparkSession


def get_spark_session(app_name: str):

    builder = (
        SparkSession.builder
         .master("local[2]")
         .appName("Reader Tests")
         .config(
             "spark.sql.extensions",
             "io.delta.sql.DeltaSparkSessionExtension",
         )
         .config(
             "spark.sql.catalog.spark_catalog",
             "org.apache.spark.sql.delta.catalog.DeltaCatalog",
         )
    )

    spark = (
        configure_spark_with_delta_pip(builder)
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")

    return spark
