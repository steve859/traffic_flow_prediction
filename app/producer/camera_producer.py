import json
import os
import random
import time
from datetime import datetime, timezone
from typing import Optional

from kafka import KafkaProducer


def _make_producer(bootstrap_servers: str) -> KafkaProducer:
    servers = [s.strip() for s in bootstrap_servers.split(",") if s.strip()]
    if not servers:
        raise ValueError("KAFKA_BOOTSTRAP_SERVERS is empty")

    return KafkaProducer(
        bootstrap_servers=servers,
        value_serializer=lambda x: json.dumps(x).encode("utf-8"),
    )


def main() -> None:
    bootstrap = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:29092")
    topic = os.getenv("KAFKA_TOPIC", "traffic_raw_data")
    interval_s = float(os.getenv("PRODUCE_INTERVAL_SECONDS", "1"))
    camera_id = os.getenv("CAMERA_ID", "CAM_01")

    producer: Optional[KafkaProducer]
    producer = _make_producer(bootstrap)
    print(f"Producer connected: {bootstrap} -> topic={topic}")

    while True:
        msg = {
            "event_id": f"evt-{random.randint(100000, 999999)}",
            "camera_id": camera_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "vehicle_type": random.choice(["motorbike", "car", "bus", "truck"]),
            "vehicle_id": random.randint(1, 10_000),
            "action": "count",
        }

        producer.send(topic, value=msg)
        producer.flush(1)
        print(f"sent: {msg}")
        time.sleep(interval_s)


if __name__ == "__main__":
    main()
