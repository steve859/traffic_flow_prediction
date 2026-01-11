import json
import os
import signal
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from kafka import KafkaConsumer, KafkaProducer

try:
    import aiohttp
except Exception:  # pragma: no cover
    aiohttp = None  # type: ignore


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _safe_json_loads(value: Any) -> Optional[dict[str, Any]]:
    if value is None:
        return None

    if isinstance(value, dict):
        return value

    if isinstance(value, (bytes, bytearray)):
        try:
            return json.loads(value.decode("utf-8"))
        except Exception:
            return None

    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return None

    return None


def _extract_camera_id(payload: dict[str, Any]) -> str:
    for key in ("camera_id", "cam_id", "camera", "cam"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "UNKNOWN"


def _extract_vehicle_type(payload: dict[str, Any]) -> str:
    value = payload.get("vehicle_type") or payload.get("type") or payload.get("class")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return "unknown"


@dataclass
class ConsumerConfig:
    bootstrap_servers: str
    topic: str
    group_id: str
    auto_offset_reset: str
    output_dir: Path
    aggregate_window_seconds: int
    forward_inference_url: str
    predictions_topic: str


def _load_config() -> ConsumerConfig:
    return ConsumerConfig(
        bootstrap_servers=os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:29092"),
        topic=os.getenv("KAFKA_TOPIC", "traffic_raw_data"),
        group_id=os.getenv("KAFKA_GROUP_ID", "traffic_consumer"),
        auto_offset_reset=os.getenv("KAFKA_AUTO_OFFSET_RESET", "earliest"),
        output_dir=Path(os.getenv("OUTPUT_DIR", "/app/data/consumer_output")),
        aggregate_window_seconds=int(os.getenv("AGGREGATE_WINDOW_SECONDS", "60")),
        forward_inference_url=os.getenv("FORWARD_INFERENCE_URL", "http://inference-api:5002/predict"),
        predictions_topic=os.getenv("KAFKA_PRED_TOPIC", "predictions"),
    )


class GracefulShutdown:
    def __init__(self) -> None:
        self.stop = False
        signal.signal(signal.SIGINT, self._handle)
        signal.signal(signal.SIGTERM, self._handle)

    def _handle(self, signum: int, frame: Any) -> None:  # noqa: ARG002
        self.stop = True


def _jsonl_append(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


async def _post_json(url: str, payload: dict[str, Any]) -> Optional[dict[str, Any]]:
    if not url:
        return None
    if aiohttp is None:
        return None

    timeout = aiohttp.ClientTimeout(total=5)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, json=payload) as resp:
                if resp.status >= 300:
                    return None
                return await resp.json()
    except Exception:
        return None


def main() -> None:
    cfg = _load_config()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    consumer = KafkaConsumer(
        cfg.topic,
        bootstrap_servers=[s.strip() for s in cfg.bootstrap_servers.split(",") if s.strip()],
        group_id=cfg.group_id,
        enable_auto_commit=True,
        auto_offset_reset=cfg.auto_offset_reset,
        value_deserializer=lambda v: v,
        consumer_timeout_ms=1000,
    )

    producer = KafkaProducer(
        bootstrap_servers=[s.strip() for s in cfg.bootstrap_servers.split(",") if s.strip()],
        value_serializer=lambda x: json.dumps(x).encode("utf-8"),
    )

    print(
        f"consumer started: topic={cfg.topic} group_id={cfg.group_id} "
        f"bootstrap={cfg.bootstrap_servers} output_dir={cfg.output_dir}"
    )

    shutdown = GracefulShutdown()

    window_start = time.time()
    counts_by_camera: dict[str, Counter[str]] = {}

    while not shutdown.stop:
        records = consumer.poll(timeout_ms=1000, max_records=500)
        for _tp, messages in records.items():
            for msg in messages:
                payload = _safe_json_loads(msg.value)
                if payload is None:
                    continue

                camera_id = _extract_camera_id(payload)
                vehicle_type = _extract_vehicle_type(payload)

                now = _utc_now()
                raw_record = {
                    "received_at": now.isoformat(),
                    "camera_id": camera_id,
                    "vehicle_type": vehicle_type,
                    "payload": payload,
                }

                raw_path = cfg.output_dir / f"raw-{now.strftime('%Y%m%d')}.jsonl"
                _jsonl_append(raw_path, raw_record)

                if camera_id not in counts_by_camera:
                    counts_by_camera[camera_id] = Counter()
                counts_by_camera[camera_id][vehicle_type] += 1

        if time.time() - window_start >= cfg.aggregate_window_seconds:
            now = _utc_now()
            for camera_id, counter in list(counts_by_camera.items()):
                if not counter:
                    continue

                agg = {
                    "camera_id": camera_id,
                    "window_seconds": cfg.aggregate_window_seconds,
                    "window_end": now.isoformat(),
                    "counts": dict(counter),
                }

                agg_path = cfg.output_dir / f"agg-{now.strftime('%Y%m%d')}.jsonl"
                _jsonl_append(agg_path, agg)

                # Optional: forward to inference-api and publish prediction result
                if cfg.forward_inference_url and aiohttp is not None:
                    try:
                        import asyncio

                        pred = asyncio.run(_post_json(cfg.forward_inference_url, agg))
                    except Exception:
                        pred = None

                    if isinstance(pred, dict):
                        pred_record = {
                            "camera_id": camera_id,
                            "window_end": now.isoformat(),
                            "input": agg,
                            "prediction": pred,
                        }
                        pred_path = cfg.output_dir / f"pred-{now.strftime('%Y%m%d')}.jsonl"
                        _jsonl_append(pred_path, pred_record)

                        try:
                            producer.send(cfg.predictions_topic, value=pred_record)
                        except Exception:
                            pass

                counts_by_camera[camera_id].clear()

            window_start = time.time()

    print("consumer stopping...")
    try:
        consumer.close(timeout=5)
    except Exception:
        pass
    try:
        producer.flush(5)
        producer.close(timeout=5)
    except Exception:
        pass


if __name__ == "__main__":
    main()
