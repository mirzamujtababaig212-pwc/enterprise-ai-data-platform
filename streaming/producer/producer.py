from __future__ import annotations
import json
import os
import random
import time
from datetime import datetime, timezone
from typing import Any
from faker import Faker
from kafka import KafkaProducer

DEFAULT_KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"
DEFAULT_KAFKA_TOPIC = "vehicle-telemetry"
DEFAULT_INTERVAL_SECONDS = 1.0


def get_kafka_bootstrap_servers() -> str:
    """Return Kafka bootstrap servers from the environment."""
    return os.getenv(
        "KAFKA_BOOTSTRAP_SERVERS",
        DEFAULT_KAFKA_BOOTSTRAP_SERVERS,
    )


def get_kafka_topic() -> str:
    """Return the Kafka telemetry topic from the environment."""
    return os.getenv(
        "KAFKA_TOPIC",
        DEFAULT_KAFKA_TOPIC,
    )


def get_interval_seconds() -> float:
    """Return the producer interval from the environment."""
    value = os.getenv(
        "KAFKA_PRODUCER_INTERVAL_SECONDS",
        str(DEFAULT_INTERVAL_SECONDS),
    )

    try:
        interval = float(value)
    except ValueError as exc:
        raise ValueError("KAFKA_PRODUCER_INTERVAL_SECONDS must be a valid number.") from exc

    if interval < 0:
        raise ValueError("KAFKA_PRODUCER_INTERVAL_SECONDS must be greater than or equal to 0.")

    return interval


def create_producer() -> KafkaProducer:
    """Create and return a configured Kafka producer."""
    return KafkaProducer(
        bootstrap_servers=get_kafka_bootstrap_servers(),
        value_serializer=lambda value: json.dumps(value).encode("utf-8"),
    )


def create_fake() -> Faker:
    """Create the Faker instance used by the producer."""
    return Faker()


def generate_vehicle_telemetry(fake: Faker | None = None) -> dict[str, Any]:
    """Generate one synthetic vehicle telemetry event."""
    del fake  # Reserved for future Faker-backed fields.

    return {
        "vehicle_id": random.choice(
            [
                "CAR001",
                "CAR002",
                "CAR003",
                "CAR004",
            ]
        ),
        "event_time": datetime.now(timezone.utc).isoformat(),
        "latitude": round(random.uniform(17.30, 17.45), 6),
        "longitude": round(random.uniform(78.35, 78.55), 6),
        "speed": round(random.uniform(20, 120), 2),
        "rpm": random.randint(800, 4500),
        "fuel_level": round(random.uniform(10, 100), 2),
        "battery": round(random.uniform(40, 100), 2),
        "engine_temperature": round(random.uniform(70, 110), 2),
        "gear": random.randint(1, 6),
    }


def publish_event(
    producer: KafkaProducer,
    message: dict[str, Any],
    *,
    topic: str | None = None,
) -> None:
    """Publish one telemetry event to Kafka."""
    producer.send(
        topic or get_kafka_topic(),
        value=message,
    )


def run_producer(
    *,
    producer: KafkaProducer | None = None,
    interval_seconds: float | None = None,
) -> None:
    """Continuously publish synthetic vehicle telemetry."""
    owns_producer = producer is None
    kafka_producer = producer or create_producer()

    interval = get_interval_seconds() if interval_seconds is None else interval_seconds

    fake = create_fake()
    topic = get_kafka_topic()

    print(
        "Producer started: "
        f"bootstrap_servers={get_kafka_bootstrap_servers()} "
        f"topic={topic} "
        f"interval_seconds={interval}"
    )

    try:
        while True:
            message = generate_vehicle_telemetry(fake)

            publish_event(
                kafka_producer,
                message,
                topic=topic,
            )

            kafka_producer.flush()

            print(message)

            if interval > 0:
                time.sleep(interval)

    except KeyboardInterrupt:
        print("\nProducer stopped.")

    finally:
        if owns_producer:
            kafka_producer.close()


def main() -> None:
    """Run the Kafka producer."""
    run_producer()


if __name__ == "__main__":
    main()
