import os
from typing import Any

from fastapi import FastAPI

try:
    from kafka import KafkaProducer
except Exception:  # pragma: no cover
    KafkaProducer = None  # type: ignore

import json

app = FastAPI(title="Traffic Inference API")

KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "")
KAFKA_PRED_TOPIC = os.getenv("KAFKA_PRED_TOPIC", "predictions")

_producer = None
if KafkaProducer and KAFKA_BOOTSTRAP_SERVERS:
    try:
        _producer = KafkaProducer(
            bootstrap_servers=[s.strip() for s in KAFKA_BOOTSTRAP_SERVERS.split(",") if s.strip()],
            value_serializer=lambda x: json.dumps(x).encode("utf-8"),
        )
    except Exception:
        _producer = None


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
async def predict(payload: dict[str, Any]) -> dict[str, Any]:
    """Minimal placeholder prediction endpoint.

    Expected payload example:
      {"camera_id": "CAM_01", "counts": {"car": 1, "motorbike": 2}}

    Returns a simple heuristic forecast and (optionally) publishes to Kafka.
    """

    counts = payload.get("counts") or {}
    try:
        total = int(sum(int(v) for v in counts.values()))
    except Exception:
        total = 0

    # Dummy heuristic: forecast next slot as same total.
    result = {
        "camera_id": payload.get("camera_id"),
        "timestamp": payload.get("timestamp"),
        "total_flow": total,
        "forecast_next": total,
    }

    if _producer:
        try:
            _producer.send(KAFKA_PRED_TOPIC, value=result)
        except Exception:
            pass

    return result
