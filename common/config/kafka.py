import os

from dotenv import load_dotenv

load_dotenv()

class KafkaConfig:
    BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS")
    TOPIC = os.getenv("KAFKA_TOPIC")

    options = {
        "kafka.bootstrap.servers": BOOTSTRAP_SERVERS,
        "subscribe": TOPIC,
        "startingOffsets": "latest"
    }
