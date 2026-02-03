import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from kafka import KafkaProducer
from PIL import Image, ImageDraw
from ultralytics import YOLO


@dataclass(frozen=True)
class YoloConfig:
    model_path: Path
    conf: float
    img_size: int
    resize_w: int
    resize_h: int
    save_annotated: bool
    annotated_dir: Path


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


def _make_producer(bootstrap_servers: str) -> KafkaProducer:
    servers = [s.strip() for s in bootstrap_servers.split(",") if s.strip()]
    if not servers:
        raise ValueError("KAFKA_BOOTSTRAP_SERVERS is empty")

    return KafkaProducer(
        bootstrap_servers=servers,
        value_serializer=lambda x: json.dumps(x, ensure_ascii=False).encode("utf-8"),
        acks="all",
        retries=5,
        request_timeout_ms=20000,
        max_block_ms=20000,
        linger_ms=50,
    )


def _connect_producer(bootstrap_servers: str) -> KafkaProducer:
    max_wait_s = _parse_float_env("KAFKA_CONNECT_MAX_WAIT_SECONDS", 60.0)
    backoff_s = _parse_float_env("KAFKA_CONNECT_RETRY_SECONDS", 2.0)

    started = time.time()
    last_err: Optional[Exception] = None
    while True:
        try:
            return _make_producer(bootstrap_servers)
        except Exception as e:
            last_err = e
            if time.time() - started > max_wait_s:
                raise
            print(f"[yolo-folder-producer] Kafka not ready yet ({e}); retrying in {backoff_s}s...")
            time.sleep(max(0.1, backoff_s))


_TIMESTAMP_PATTERNS: List[re.Pattern] = [
    # 2026-01-12T13:05:00 or 2026-01-12 13:05:00
    re.compile(r"(\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2})"),
    # 20260112_130500 or 20260112-130500
    re.compile(r"(\d{8})[_-](\d{6})"),
    # 20260112130500
    re.compile(r"(\d{14})"),
]


def _try_parse_timestamp(text: str) -> Optional[str]:
    s = (text or "").strip()
    if not s:
        return None

    for pat in _TIMESTAMP_PATTERNS:
        m = pat.search(s)
        if not m:
            continue

        try:
            if len(m.groups()) == 1:
                token = m.group(1)
                # normalize space -> T
                token = token.replace(" ", "T")
                dt = datetime.fromisoformat(token)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt.isoformat()

            if len(m.groups()) == 2:
                d, t = m.group(1), m.group(2)
                dt = datetime.strptime(f"{d}{t}", "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
                return dt.isoformat()

        except Exception:
            continue

    # 14 digits pattern group
    m = _TIMESTAMP_PATTERNS[-1].search(s)
    if m:
        try:
            dt = datetime.strptime(m.group(1), "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
            return dt.isoformat()
        except Exception:
            return None

    return None


# NOTE: Don't use word-boundary (\b) here because '_' is a word char.
# Filenames like 'cam1_20260113_022630.jpg' should still match.
_CAM_RE = re.compile(r"(cam\d+)", re.IGNORECASE)


def _infer_camera_id(path: Path) -> Optional[str]:
    m = _CAM_RE.search(path.stem)
    if m:
        return m.group(1).lower()

    # fallback: parent folder name might be cam1/...
    m2 = _CAM_RE.search(path.parent.name)
    if m2:
        return m2.group(1).lower()

    return None


def _iter_groups(data_dir: Path) -> Iterable[Tuple[str, List[Path]]]:
    """Yield (group_key, image_paths).

    Supported layouts:
    1) data_to_demo/<group>/cam1.jpg..cam8.jpg
    2) data_to_demo/*.jpg with timestamps in filenames (grouped by timestamp token)
    """

    subdirs = sorted([p for p in data_dir.iterdir() if p.is_dir()]) if data_dir.exists() else []
    if subdirs:
        for d in subdirs:
            imgs = sorted([p for p in d.glob("*.*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
            if imgs:
                yield d.name, imgs
        return

    imgs = sorted([p for p in data_dir.glob("*.*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    if not imgs:
        return

    # group by extracted timestamp token; if none, each image is its own group
    groups: Dict[str, List[Path]] = {}
    for p in imgs:
        ts = _try_parse_timestamp(p.name) or "__no_ts__"
        groups.setdefault(ts, []).append(p)

    for k in sorted(groups.keys()):
        yield k, sorted(groups[k])


def _count_from_boxes(
    boxes, names: Dict[int, str], class_map: Dict[str, List[str]]
) -> Tuple[Dict[str, int], int]:
    # Initialize with the canonical 4 types expected by downstream code.
    counts: Dict[str, int] = {"motorbike": 0, "car": 0, "truck": 0, "bus": 0}

    # reverse map for faster lookup
    label_to_key: Dict[str, str] = {}
    for k, aliases in class_map.items():
        for a in aliases:
            label_to_key[a.lower()] = k

    if boxes is None:
        return counts, 0

    cls_arr = boxes.cls
    if cls_arr is None:
        return counts, 0

    cls_ids = cls_arr.cpu().numpy().astype(int).tolist()
    for cls_id in cls_ids:
        label = str(names.get(int(cls_id), str(cls_id))).lower()
        key = label_to_key.get(label)
        if key is None:
            # unknown class -> ignore (keeps schema stable)
            continue
        counts[key] += 1

    total = sum(counts.values())
    return counts, total


def _annotate_image(img_rgb: np.ndarray, results0, names: Dict[int, str]) -> Image.Image:
    im = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(im)

    boxes = results0.boxes
    if boxes is None:
        return im

    for b in boxes:
        x1, y1, x2, y2 = map(int, b.xyxy[0])
        cls_id = int(b.cls[0])
        conf = float(b.conf[0])
        label = f"{names.get(cls_id, str(cls_id))} {conf:.2f}"
        draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=2)
        draw.text((x1, max(0, y1 - 12)), label, fill=(0, 255, 0))

    return im


def main() -> None:
    bootstrap = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:29092")
    topic = os.getenv("KAFKA_TOPIC", "traffic_counts_raw")

    data_dir = Path(os.getenv("DEMO_DATA_DIR", "/app/data_to_demo"))
    if not data_dir.exists():
        raise FileNotFoundError(f"DEMO_DATA_DIR not found: {data_dir}")

    # YOLO config
    model_path = Path(os.getenv("YOLO_MODEL_PATH", "/app/vision/detector/yolov8n_multiclass_best.pt"))
    conf = _parse_float_env("YOLO_CONF", 0.25)
    img_size = _parse_int_env("YOLO_IMGSZ", 640)
    resize_w = _parse_int_env("YOLO_RESIZE_W", 640)
    resize_h = _parse_int_env("YOLO_RESIZE_H", 384)

    save_annotated = os.getenv("YOLO_SAVE_ANNOTATED", "false").lower() in {"1", "true", "yes"}
    annotated_dir = Path(os.getenv("YOLO_ANNOTATED_DIR", "/app/data/yolo_annotated"))

    cfg = YoloConfig(
        model_path=model_path,
        conf=conf,
        img_size=img_size,
        resize_w=resize_w,
        resize_h=resize_h,
        save_annotated=save_annotated,
        annotated_dir=annotated_dir,
    )

    if not cfg.model_path.exists():
        raise FileNotFoundError(f"YOLO model not found: {cfg.model_path}")

    # class mapping (custom model labels -> canonical keys)
    # You can override via env YOLO_CLASS_MAP_JSON.
    default_map = {
        "motorbike": ["motorbike", "motorcycle", "bike"],
        "car": ["car"],
        "truck": ["truck"],
        "bus": ["bus"],
    }
    class_map = default_map
    if os.getenv("YOLO_CLASS_MAP_JSON"):
        class_map = json.loads(os.getenv("YOLO_CLASS_MAP_JSON", "{}"))

    # pacing
    interval_s = _parse_float_env("DEMO_GROUP_INTERVAL_SECONDS", 1.0)
    loop_forever = os.getenv("DEMO_LOOP", "false").lower() in {"1", "true", "yes"}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[yolo-folder-producer] device={device}")
    print(f"[yolo-folder-producer] data_dir={data_dir}")
    print(f"[yolo-folder-producer] model={cfg.model_path}")
    print(f"[yolo-folder-producer] topic={topic} bootstrap={bootstrap}")

    model = YOLO(str(cfg.model_path))
    model.to(device)
    names = model.names or {}
    print(f"[yolo-folder-producer] model classes={names}")

    producer = _connect_producer(bootstrap)

    def run_once() -> None:
        groups = list(_iter_groups(data_dir))
        if not groups:
            print(f"[yolo-folder-producer] No images found under {data_dir}")
            return

        print(f"[yolo-folder-producer] found_groups={len(groups)}")

        for group_key, img_paths in groups:
            # Try to extract a shared timestamp from group key; fallback to now
            ts_iso = _try_parse_timestamp(group_key) or datetime.now(timezone.utc).isoformat()

            sent = 0
            first_ack = None
            for img_path in img_paths:
                cam = _infer_camera_id(img_path)
                if not cam:
                    # Skip images that don't declare camera id
                    print(f"[yolo-folder-producer] skip_no_camera_id: {img_path}")
                    continue

                try:
                    im = Image.open(img_path).convert("RGB")
                except Exception:
                    continue

                if cfg.resize_w > 0 and cfg.resize_h > 0:
                    im = im.resize((cfg.resize_w, cfg.resize_h))

                img_rgb = np.asarray(im)
                results = model(img_rgb, conf=cfg.conf, imgsz=cfg.img_size, verbose=False)
                r0 = results[0]

                counts, total = _count_from_boxes(r0.boxes, names=names, class_map=class_map)

                payload = {
                    "camera_id": cam,
                    "timestamp": ts_iso,
                    "counts": counts,
                    "total": int(total),
                    "source": "yolo_folder",
                    "meta": {
                        "group": group_key,
                        "image": img_path.name,
                    },
                }

                try:
                    # Force an ack (and surface errors) so we don't "silently succeed".
                    future = producer.send(topic, value=payload)
                    ack = future.get(timeout=10)
                    if first_ack is None:
                        first_ack = ack
                    sent += 1
                except Exception as e:
                    print(f"[yolo-folder-producer] kafka_send_failed cam={cam} image={img_path.name} err={e!r}")
                    continue

                if cfg.save_annotated:
                    cfg.annotated_dir.mkdir(parents=True, exist_ok=True)
                    annotated = _annotate_image(img_rgb, r0, names=names)
                    out_name = f"{group_key}_{cam}_{img_path.name}"
                    annotated.save(str(cfg.annotated_dir / out_name))

            producer.flush(10)
            if first_ack is not None:
                print(
                    f"[yolo-folder-producer] group={group_key} ts={ts_iso} sent={sent} "
                    f"first_ack={{topic={first_ack.topic}, partition={first_ack.partition}, offset={first_ack.offset}}}"
                )
            else:
                print(f"[yolo-folder-producer] group={group_key} ts={ts_iso} sent={sent}")
            time.sleep(max(0.0, interval_s))

    while True:
        run_once()
        if not loop_forever:
            break
        time.sleep(1.0)


if __name__ == "__main__":
    main()
