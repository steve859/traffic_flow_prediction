import argparse
import json
import os
from typing import Any, Dict, Optional

from kafka import KafkaConsumer


def _try_parse_json(s: str) -> Optional[Dict[str, Any]]:
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
        return {"_": obj}
    except Exception:
        return None


def main() -> None:
    p = argparse.ArgumentParser(description="Peek N messages from a Kafka topic (debug helper).")
    p.add_argument("--bootstrap", default=os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:29092"))
    p.add_argument("--topic", default=os.getenv("KAFKA_TOPIC", "traffic_counts_raw"))
    p.add_argument("--max", type=int, default=5, help="Max messages to print")
    p.add_argument(
        "--source",
        default=None,
        help="If set, only print messages whose JSON has source==this value (e.g. yolo_folder)",
    )
    p.add_argument(
        "--from-beginning",
        action="store_true",
        help="Read from earliest offset (default is latest)",
    )
    p.add_argument("--timeout-ms", type=int, default=5000, help="Stop if no messages in this time")
    args = p.parse_args()

    consumer = KafkaConsumer(
        args.topic,
        bootstrap_servers=[s.strip() for s in args.bootstrap.split(",") if s.strip()],
        auto_offset_reset="earliest" if args.from_beginning else "latest",
        enable_auto_commit=False,
        consumer_timeout_ms=args.timeout_ms,
        value_deserializer=lambda b: b.decode("utf-8", errors="replace"),
    )

    count = 0
    for msg in consumer:
        val = msg.value
        parsed = _try_parse_json(val)

        if args.source is not None:
            if parsed is None or str(parsed.get("source")) != args.source:
                continue

        count += 1
        if parsed is not None:
            print(json.dumps(parsed, ensure_ascii=False))
        else:
            print(val)

        if count >= args.max:
            break

    print(f"printed={count}")


if __name__ == "__main__":
    main()
