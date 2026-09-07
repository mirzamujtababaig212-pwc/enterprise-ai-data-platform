from common.config.settings import Settings
from common.spark.spark_builder import SparkSessionBuilder


def main():
    spark = SparkSessionBuilder.build("ReadBronze")

    try:
        df = spark.read.format("delta").load(Settings.storage.BRONZE_PATH)

        print("BRONZE ROW COUNT:", df.count())

        print("\nBRONZE COLUMNS:")
        for column in df.columns:
            print(column)

        print("\nBRONZE SCHEMA:")
        df.printSchema()

        print("\nSAMPLE DATA:")
        df.show(5, truncate=False)

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
