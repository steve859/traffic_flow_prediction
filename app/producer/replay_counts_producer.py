import csv
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from kafka import KafkaProducer


def _make_producer(bootstrap_servers: str) -> KafkaProducer:
    servers = [s.strip() for s in bootstrap_servers.split(",") if s.strip()]
    if not servers:
        raise ValueError("KAFKA_BOOTSTRAP_SERVERS is empty")

    return KafkaProducer(
        bootstrap_servers=servers,
        value_serializer=lambda x: json.dumps(x, ensure_ascii=False).encode("utf-8"),
    )


def _parse_float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default


def _parse_int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def main() -> None:
    bootstrap = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:29092")
    topic = os.getenv("KAFKA_TOPIC", "traffic_counts_raw")

    csv_path = Path(os.getenv("COUNTS_CSV", "/app/hcmc_flow.csv"))
    speedup = _parse_float_env("REPLAY_SPEEDUP", 60.0)  # higher => faster replay
    loop_forever = os.getenv("REPLAY_LOOP", "true").lower() in {"1", "true", "yes"}
    max_rows = _parse_int_env("REPLAY_MAX_ROWS", 0)  # 0 = no limit

    # For demos, it can be useful to emit timestamps close to "now" so downstream UI looks realtime.
    # Modes:
    # - original (default): use timestamp_vn from CSV and sleep based on its deltas (scaled by speedup)
    # - now: override timestamp with current time; sleep a fixed amount per message
    ts_mode = os.getenv("REPLAY_TIMESTAMP_MODE", "original").strip().lower()
    sleep_s = _parse_float_env("REPLAY_SLEEP_SECONDS", 0.2)

    if not csv_path.exists():
        raise FileNotFoundError(f"COUNTS_CSV not found: {csv_path}")

    producer: Optional[KafkaProducer] = None
    max_wait_s = _parse_float_env("KAFKA_CONNECT_MAX_WAIT_SECONDS", 60.0)
    backoff_s = _parse_float_env("KAFKA_CONNECT_RETRY_SECONDS", 2.0)

    started = time.time()
    while producer is None:
        try:
            producer = _make_producer(bootstrap)
        except Exception as e:
            if time.time() - started > max_wait_s:
                raise
            print(f"Kafka not ready yet ({e}); retrying in {backoff_s}s...")
            time.sleep(backoff_s)

    print(f"Replay producer connected: {bootstrap} -> topic={topic}")
    print(f"Reading: {csv_path} speedup={speedup} loop={loop_forever} ts_mode={ts_mode}")

    while True:
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)

            prev_ts: Optional[datetime] = None
            sent = 0

            for row in reader:
                if max_rows and sent >= max_rows:
                    break

                cam = (row.get("camera_id") or "").strip() or "UNKNOWN"
                ts_str = (row.get("timestamp_vn") or "").strip()
                if ts_mode == "now":
                    ts_str = datetime.now().isoformat(timespec="seconds")
                try:
                    ts = datetime.fromisoformat(ts_str)
                except Exception:
                    # fallback: send as-is
                    ts = None

                counts = {
                    "motorbike": int(float(row.get("motorbike") or 0)),
                    "car": int(float(row.get("car") or 0)),
                    "truck": int(float(row.get("truck") or 0)),
                    "bus": int(float(row.get("bus") or 0)),
                }
                total = int(float(row.get("total") or sum(counts.values())))

                payload: dict[str, Any] = {
                    "camera_id": cam,
                    # Keep timestamp as ISO string for Spark
                    "timestamp": ts_str,
                    "counts": counts,
                    "total": total,
                    "source": "csv_replay",
                }

                if ts_mode == "now":
                    if sleep_s > 0:
                        time.sleep(sleep_s)
                else:
                    # Sleep according to original timestamp deltas (scaled by speedup)
                    if ts is not None and prev_ts is not None:
                        delta = (ts - prev_ts).total_seconds()
                        if delta > 0:
                            time.sleep(delta / max(speedup, 1e-6))
                    if ts is not None:
                        prev_ts = ts

                producer.send(topic, value=payload)
                sent += 1

                if sent % 200 == 0:
                    producer.flush(5)
                    print(f"sent {sent} messages, last={cam} ts={ts_str} total={total}")

        producer.flush(5)
        print("Reached end of CSV")
        if not loop_forever:
            break


if __name__ == "__main__":
    main()
