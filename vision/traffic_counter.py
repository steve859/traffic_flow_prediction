import cv2
import numpy as np
import json
import uuid
import time
from datetime import datetime
from ultralytics import YOLO
from kafka import KafkaProducer

# Import config từ file cùng thư mục
try:
    from config import VIDEO_CONFIGS, MODEL_PATH, TARGET_SIZE, CONF_THRESHOLD, CLASS_IDS, KAFKA_TOPIC, KAFKA_SERVER
except ImportError:
    # Fallback nếu chạy từ root
    from src.vision.config import VIDEO_CONFIGS, MODEL_PATH, TARGET_SIZE, CONF_THRESHOLD, CLASS_IDS, KAFKA_TOPIC, KAFKA_SERVER

# --- CỜ BẬT/TẮT TÍNH NĂNG ---
ENABLE_KAFKA = False  # Set = True khi bạn đã chạy Docker Kafka
ENABLE_SIDE_BY_SIDE = True # Set = True để xem 2 màn hình (Gốc vs AI)

# --- KHỞI TẠO KAFKA ---
producer = None
if ENABLE_KAFKA:
    try:
        producer = KafkaProducer(
            bootstrap_servers=[KAFKA_SERVER],
            value_serializer=lambda x: json.dumps(x).encode('utf-8')
        )
        print(f"✅ Đã kết nối Kafka tại {KAFKA_SERVER}")
    except Exception as e:
        print(f"⚠️ Không thể kết nối Kafka: {e}. Đang chạy chế độ Offline.")
        ENABLE_KAFKA = False

# --- HÀM TOÁN HỌC (CROSS PRODUCT) ---
def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

def intersect(A, B, C, D):
    """Kiểm tra vector AB có cắt đoạn thẳng CD không"""
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

def scale_line_coords(line, original_size, target_size):
    """Scale tọa độ dòng kẻ theo tỷ lệ resize"""
    orig_w, orig_h = original_size
    target_w, target_h = target_size
    x_scale = target_w / orig_w
    y_scale = target_h / orig_h
    start = (int(line['start'][0] * x_scale), int(line['start'][1] * y_scale))
    end = (int(line['end'][0] * x_scale), int(line['end'][1] * y_scale))
    return start, end

# --- HÀM XỬ LÝ CHÍNH ---
def process_camera(cam_id):
    config = VIDEO_CONFIGS.get(cam_id)
    if not config:
        print(f"❌ Error: Không tìm thấy config cho {cam_id}")
        return

    print(f"▶️ Bắt đầu xử lý: {cam_id}")
    cap = cv2.VideoCapture(config['path'])
    model = YOLO(MODEL_PATH)

    # Lấy thông số video gốc
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Scale tọa độ lines
    scaled_lines = []
    for line in config['lines']:
        s, e = scale_line_coords(line, (orig_w, orig_h), TARGET_SIZE)
        scaled_lines.append({
            "name": line['name'], "start": s, "end": e, "count": 0
        })

    track_history = {}
    counted_ids = set() # Tránh đếm trùng

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Video ended.")
            break

        # 1. Resize
        frame = cv2.resize(frame, TARGET_SIZE)
        
        # 2. Tạo bản sao cho video gốc (Nếu bật chế độ xem 2 màn hình)
        if ENABLE_SIDE_BY_SIDE:
            raw_view = frame.copy()
            cv2.putText(raw_view, "ORIGINAL INPUT", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # 3. Tracking YOLO
        results = model.track(frame, persist=True, conf=CONF_THRESHOLD, classes=CLASS_IDS, verbose=False)
        
        # 4. Vẽ Line lên màn hình AI
        for line in scaled_lines:
            cv2.line(frame, line['start'], line['end'], (0, 255, 255), 2)
            cv2.putText(frame, f"{line['name']}: {line['count']}", 
                        (line['start'][0], line['start'][1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # 5. Xử lý logic đếm xe
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            class_ids = results[0].boxes.cls.int().cpu().tolist()

            for box, track_id, cls in zip(boxes, track_ids, class_ids):
                x, y, w, h = box
                center = (int(x), int(y))
                cls_name = model.names[cls]

                # Kiểm tra giao cắt
                if track_id in track_history:
                    prev_center = track_history[track_id]
                    for line in scaled_lines:
                        count_key = f"{line['name']}_{track_id}"
                        
                        if count_key not in counted_ids:
                            if intersect(prev_center, center, line['start'], line['end']):
                                # --- ĐẾM THÀNH CÔNG ---
                                line['count'] += 1
                                counted_ids.add(count_key)
                                
                                # Gửi Kafka
                                msg = {
                                    "event_id": str(uuid.uuid4()),
                                    "camera_id": cam_id,
                                    "timestamp": datetime.now().isoformat(),
                                    "vehicle_id": track_id,
                                    "vehicle_type": cls_name,
                                    "action": "cross_line"
                                }
                                if ENABLE_KAFKA and producer:
                                    producer.send(KAFKA_TOPIC, value=msg)
                                    print(f"📡 Sent Kafka: {msg}")
                                else:
                                    print(f"✅ Counted: {cls_name} (Total: {line['count']})")

                                # Visual Effect
                                cv2.circle(frame, center, 15, (0, 0, 255), -1)

                track_history[track_id] = center
                
                # Vẽ Box
                cv2.rectangle(frame, (int(x-w/2), int(y-h/2)), (int(x+w/2), int(y+h/2)), (0, 255, 0), 2)
                cv2.putText(frame, f"{track_id}", (int(x), int(y)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.putText(frame, "AI PROCESSING", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # 6. Hiển thị
        final_view = frame
        if ENABLE_SIDE_BY_SIDE:
            final_view = np.hstack((raw_view, frame))

        cv2.imshow(f"Traffic Monitor - {cam_id}", final_view)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Chọn camera để chạy
    process_camera("CAM_01") 
    
    # process_camera("CAM_04") # Chạy cam khác