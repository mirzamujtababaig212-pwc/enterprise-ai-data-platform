import json
import random
import time
from datetime import datetime

from faker import Faker
from kafka import KafkaProducer

producer = KafkaProducer(
    bootstrap_servers="localhost:9092",
    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
)

fake = Faker()

print("Producer started...\n")

while True:
    message = {
        "vehicle_id": random.choice(["CAR001", "CAR002", "CAR003", "CAR004"]),
        "event_time": datetime.now().isoformat(),
        "latitude": round(random.uniform(17.30, 17.45), 6),
        "longitude": round(random.uniform(78.35, 78.55), 6),
        "speed": round(random.uniform(20, 120), 2),
        "rpm": random.randint(800, 4500),
        "fuel_level": round(random.uniform(10, 100), 2),
        "battery": round(random.uniform(40, 100), 2),
        "engine_temperature": round(random.uniform(70, 110), 2),
        "gear": random.randint(1, 6),
    }

    producer.send("vehicle-telemetry", message)

    producer.flush()

    print(message)

    time.sleep(1)
