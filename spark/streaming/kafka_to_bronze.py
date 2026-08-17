from common.config.settings import Settings
from common.logging.logger import get_logger
from common.readers.kafka_reader import KafkaReader
from common.spark.spark_builder import SparkSessionBuilder
from common.transformers.bronze_transformer import BronzeTransformer

logger = get_logger(__name__)


def main():

    print("=" * 80)
    print("KAFKA → BRONZE STREAMING PIPELINE")
    print("=" * 80)

    print(f"Kafka bootstrap servers : " f"{Settings.kafka.BOOTSTRAP_SERVERS}")

    print(f"Kafka topic             : " f"{Settings.kafka.TOPIC}")

    print(f"Bronze table             : " f"{Settings.storage.BRONZE_TABLE}")

    print(f"Bronze path              : " f"{Settings.storage.BRONZE_PATH}")

    print(f"Checkpoint               : " f"{Settings.storage.BRONZE_CHECKPOINT}")

    spark = SparkSessionBuilder.build("KafkaToBronzeStreaming")

    try:

        # ----------------------------------------------------------
        # Create databases
        # ----------------------------------------------------------

        spark.sql("CREATE DATABASE IF NOT EXISTS bronze")

        # ----------------------------------------------------------
        # Kafka reader
        # ----------------------------------------------------------

        reader = KafkaReader(Settings.kafka.options)

        kafka_df = reader.read(spark)

        # ----------------------------------------------------------
        # Bronze transformation
        # ----------------------------------------------------------

        bronze_df = BronzeTransformer.transform(kafka_df)

        # ----------------------------------------------------------
        # Streaming write
        # ----------------------------------------------------------

        query = (
            bronze_df.writeStream.format("delta")
            .outputMode("append")
            .option("mergeSchema", "true")
            .option(
                "checkpointLocation",
                Settings.storage.BRONZE_CHECKPOINT,
            )
            .option(
                "path",
                Settings.storage.BRONZE_PATH,
            )
            .toTable(Settings.storage.BRONZE_TABLE)
        )

        logger.info("Kafka → Bronze streaming query started")

        print()
        print("=" * 80)
        print("STREAMING QUERY STARTED")
        print("=" * 80)
        print()
        print(f"Kafka topic : {Settings.kafka.TOPIC}")
        print(f"Bronze table: {Settings.storage.BRONZE_TABLE}")
        print()
        print("Waiting for Kafka events...")
        print("Press Ctrl+C to stop.")
        print()

        query.awaitTermination()

    except KeyboardInterrupt:

        logger.info("Stopping Kafka → Bronze streaming pipeline")

    finally:

        spark.stop()


if __name__ == "__main__":
    main()
