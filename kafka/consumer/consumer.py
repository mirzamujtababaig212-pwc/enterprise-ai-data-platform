import json

import psycopg2

from kafka import KafkaConsumer

consumer = KafkaConsumer(
    "vehicle-telemetry",
    bootstrap_servers="localhost:9092",
    auto_offset_reset="earliest",
    enable_auto_commit=True,
    value_deserializer=lambda x: json.loads(x.decode("utf-8")),
)

conn = psycopg2.connect(
    host="localhost",
    port=5432,
    database="vehicle_platform",
    user="postgres",
    password="password",
)

cursor = conn.cursor()

print("Consumer started...\n")

for msg in consumer:

    data = msg.value

    cursor.execute(
        """
        INSERT INTO streaming.vehicle_telemetry
        (
            vehicle_id,
            event_time,
            latitude,
            longitude,
            speed,
            rpm,
            fuel_level,
            battery,
            engine_temperature,
            gear
        )
        VALUES
        (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """,
        (
            data["vehicle_id"],
            data["event_time"],
            data["latitude"],
            data["longitude"],
            data["speed"],
            data["rpm"],
            data["fuel_level"],
            data["battery"],
            data["engine_temperature"],
            data["gear"],
        ),
    )

    conn.commit()

    print(data)
