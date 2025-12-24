"""
Test custom YOLOv8 model (best.pt) on traffic camera images
- Works with custom-trained classes
- Counts all detected objects as vehicles
"""

import os
from pathlib import Path
import cv2
import torch
from ultralytics import YOLO

# =========================
# CONFIG (WINDOWS PATHS)
# =========================

# Path to custom YOLO model
MODEL_PATH = r"D:\Code\traffic_flow_prediction\vision\detector\yolov8s.pt"

# Path to one camera folder
IMAGE_DIR = r"D:\Code\traffic_flow_prediction\data\cam1"

# Output folder
OUTPUT_DIR = r"D:\Code\traffic_flow_prediction\yolov8_test_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Resize before inference (important for low-res cameras)
RESIZE_WIDTH = 640
RESIZE_HEIGHT = 384

# Confidence threshold (custom model → có thể thấp hơn)
CONF_THRES = 0.25

# =========================
# LOAD MODEL
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device: {device}")

model = YOLO(MODEL_PATH)
model.to(device)

print("[INFO] Model loaded successfully")
print("[INFO] Model classes:", model.names)

# =========================
# LOAD IMAGES
# =========================
image_paths = sorted(Path(IMAGE_DIR).glob("*.jpg"))

if not image_paths:
    raise RuntimeError(f"No images found in {IMAGE_DIR}")

print(f"[INFO] Found {len(image_paths)} images")

# =========================
# RUN INFERENCE
# =========================
for img_path in image_paths[:10]:  # test 10 ảnh đầu
    img = cv2.imread(str(img_path))
    if img is None:
        continue

    # Resize for YOLO
    img_resized = cv2.resize(img, (RESIZE_WIDTH, RESIZE_HEIGHT))

    # Inference
    results = model(
        img_resized,
        conf=CONF_THRES,
        verbose=False
    )

    boxes = results[0].boxes
    vehicle_count = 0

    if boxes is not None:
        vehicle_count = len(boxes)

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            class_name = model.names.get(cls_id, str(cls_id))
            label = f"{class_name} {conf:.2f}"

            cv2.rectangle(img_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                img_resized,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1
            )

    print(f"[RESULT] {img_path.name}: vehicle_count = {vehicle_count}")

    # Save output
    out_path = Path(OUTPUT_DIR) / img_path.name
    cv2.imwrite(str(out_path), img_resized)

print(f"\n[DONE] Results saved to: {OUTPUT_DIR}")
