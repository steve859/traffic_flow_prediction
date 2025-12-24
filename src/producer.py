import cv2
import json
import time
import numpy as np
from ultralytics import YOLO
from kafka import KafkaProducer
from datetime import datetime
import uuid

# --- CẤU HÌNH ---
ENABLE_KAFKA = True  # Đổi thành False nếu chỉ muốn test Video/CV mà không gửi Kafka
KAFKA_TOPIC = 'traffic_raw_data'
KAFKA_SERVER = 'localhost:9092' 

# Configs từ bạn (Đã chuẩn hóa key)
VIDEO_CONFIGS = { 
    "CAM_01":{
        "path": "data/processed_videos/cam01_CongHoaUtTich1.mp4",
        "lines": [{"name": "line1", "start": (69, 388), "end": (559, 389)}]
    },
    "CAM_02":{
        "path": "data/processed_videos/cam02_CongHoaUtTich1.mp4",
        "lines": [{"name": "line1", "start": (70, 454), "end": (473, 463)}]
    },
    # ... (Điền tiếp các cam khác của bạn vào đây) ...
}

MODEL_PATH = "yolov8s.pt"
TARGET_SIZE = (640, 640) # Lưu ý: Tọa độ Line phải khớp với size này

# --- KHỞI TẠO ---
model = YOLO(MODEL_PATH)
producer = None

if ENABLE_KAFKA:
    try:
        producer = KafkaProducer(
            bootstrap_servers=[KAFKA_SERVER],
            value_serializer=lambda x: json.dumps(x).encode('utf-8')
        )
        print(f"✅ Đã kết nối Kafka tại {KAFKA_SERVER}")
    except Exception as e:
        print(f"❌ Lỗi kết nối Kafka: {e}")
        ENABLE_KAFKA = False

# --- HÀM HỖ TRỢ ---
def is_crossed(p1, p2, line_start, line_end):
    """
    Kiểm tra vector chuyển động (p1 -> p2) có cắt qua đoạn thẳng (line_start -> line_end) không.
    Sử dụng thuật toán Cross Product (tích có hướng).
    """
    p1 = np.array(p1) # Vị trí cũ
    p2 = np.array(p2) # Vị trí mới
    l1 = np.array(line_start)
    l2 = np.array(line_end)

    # Hàm tính hướng (ccw)
    def ccw(A, B, C):
        return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

    # 2 đoạn thẳng cắt nhau khi CCW đảo chiều
    return ccw(p1, l1, l2) != ccw(p2, l1, l2) and ccw(p1, p2, l1) != ccw(p1, p2, l2)

# --- MAIN LOOP ---
def process_video(cam_id, config):
    cap = cv2.VideoCapture(config["path"])
    lines = config["lines"]
    
    # Dictionary lưu vị trí cũ của các xe: {track_id: (x, y)}
    track_history = {} 
    
    # Tạo biến đếm local để hiển thị lên màn hình
    counter = {line['name']: 0 for line in lines}

    print(f"▶️ Đang xử lý: {cam_id}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 1. Resize về đúng kích thước đã chọn tọa độ (640x640)
        frame = cv2.resize(frame, TARGET_SIZE)
        
        # 2. YOLOv8 Tracking (persist=True để giữ ID qua các frame)
        # classes=[2,3,5,7] thường là Car, Motorcycle, Bus, Truck trong COCO dataset
        # Tuy nhiên YOLOv8 COCO: 2=car, 3=motorcycle, 5=bus, 7=truck. Hãy check lại model của bạn.
        results = model.track(frame, persist=True, verbose=False, classes=[2, 3, 5, 7]) 
        
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu() # x_center, y_center, w, h
            track_ids = results[0].boxes.id.int().cpu().tolist()
            class_ids = results[0].boxes.cls.int().cpu().tolist()

            for box, track_id, cls in zip(boxes, track_ids, class_ids):
                x, y, w, h = box
                center = (int(x), int(y))
                cls_name = model.names[cls]

                # 3. Kiểm tra Crossing Logic
                if track_id in track_history:
                    prev_center = track_history[track_id]
                    
                    for line in lines:
                        # Kiểm tra xem xe có cắt qua line không
                        if is_crossed(prev_center, center, line['start'], line['end']):
                            
                            # Tăng biến đếm hiển thị
                            counter[line['name']] += 1
                            
                            # Tạo message gửi Kafka
                            msg = {
                                "camera_id": cam_id,
                                "timestamp": datetime.now().isoformat(), # Thời gian thực lúc chạy
                                "line_id": line['name'],
                                "vehicle_id": track_id,
                                "vehicle_type": cls_name,
                                "event_id": str(uuid.uuid4())
                            }
                            
                            print(f"🚀 Sent Kafka: {msg}")
                            
                            if ENABLE_KAFKA and producer:
                                producer.send(KAFKA_TOPIC, value=msg)

                            # Visual effect: Vẽ chấm đỏ khi cắt qua
                            cv2.circle(frame, center, 10, (0, 0, 255), -1)

                # Cập nhật vị trí mới
                track_history[track_id] = center

                # Vẽ Bbox và ID
                cv2.rectangle(frame, (int(x-w/2), int(y-h/2)), (int(x+w/2), int(y+h/2)), (0, 255, 0), 2)
                cv2.putText(frame, f"{track_id}-{cls_name}", (int(x-w/2), int(y-h/2)-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 4. Vẽ Line và Số đếm lên màn hình
        for line in lines:
            cv2.line(frame, line['start'], line['end'], (0, 255, 255), 2)
            cv2.putText(frame, f"{line['name']}: {counter[line['name']]}", 
                        (line['start'][0], line['start'][1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow(f"Processing {cam_id}", frame)

        # Nhấn 'q' để thoát
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# --- CHẠY CHƯƠNG TRÌNH ---
if __name__ == "__main__":
    # Chọn chạy thử 1 camera trước
    process_video("CAM_01", VIDEO_CONFIGS["CAM_01"])
    
    # Nếu muốn chạy hết list:
    # for cam_id, config in VIDEO_CONFIGS.items():
    #     process_video(cam_id, config)