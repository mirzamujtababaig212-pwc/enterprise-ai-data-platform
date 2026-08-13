import os

from dotenv import load_dotenv

load_dotenv()


class KafkaConfig:

    BOOTSTRAP_SERVERS = os.getenv(
        "KAFKA_BOOTSTRAP_SERVERS",
        "localhost:9092",
    )

    TOPIC = os.getenv(
        "KAFKA_TOPIC",
        "vehicle-telemetry",
    )

    STARTING_OFFSETS = os.getenv(
        "KAFKA_STARTING_OFFSETS",
        "latest",
    )

    FAIL_ON_DATA_LOSS = os.getenv(
        "KAFKA_FAIL_ON_DATA_LOSS",
        "false",
    )

    options = {
        "kafka.bootstrap.servers": BOOTSTRAP_SERVERS,
        "subscribe": TOPIC,
        "startingOffsets": STARTING_OFFSETS,
        "failOnDataLoss": FAIL_ON_DATA_LOSS,
    }
