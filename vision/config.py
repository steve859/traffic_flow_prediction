import os

# --- CẤU HÌNH ĐƯỜNG DẪN TỰ ĐỘNG ---
# Lấy đường dẫn thư mục hiện tại (src/vision)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Lấy đường dẫn root project (D:/Code/traffic_flow_prediction)
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))

def get_path(relative_path):
    """Chuyển đường dẫn tương đối thành tuyệt đối dựa trên Project Root"""
    return os.path.join(PROJECT_ROOT, relative_path)

# --- CẤU HÌNH VIDEO & LINE ---
VIDEO_CONFIGS = {
    "CAM_01": {
        "path": get_path("data/processed_videos/cam01_CongHoaUtTich1.mp4"),
        "lines": [
            {"name": "line1", "start": (69, 388), "end": (559, 389)}
        ]
    },
    "CAM_02": {
        "path": get_path("data/processed_videos/cam02_CongHoaUtTich1.mp4"),
        "lines": [
            {"name": "line1", "start": (70, 454), "end": (473, 463)}
        ]
    },
    "CAM_03": {
        "path": get_path("data/processed_videos/cam03_DuongBaTrac_TaQuangBuu.mp4"),
        "lines": [
            {"name": "line1", "start": (207, 456), "end": (629, 370)}
        ]
    },
    "CAM_04": {
        "path": get_path("data/processed_videos/cam04_HamQ1.mp4"),
        "lines": [
            {"name": "line1", "start": (43, 319), "end": (247, 324)},
            {"name": "line2", "start": (299, 283), "end": (500, 265)}
        ]
    },
    "CAM_05": {
        "path": get_path("data/processed_videos/cam05_NKKN_VoThiSau.mp4"),
        "lines": [
            {"name": "line1", "start": (30, 335), "end": (434, 308)}
        ]
    },
    "CAM_06": {
        "path": get_path("data/processed_videos/cam06_D2_UngVanKhiem.mp4"),
        "lines": [
            {"name": "line1", "start": (15, 393), "end": (311, 296)},
            {"name": "line2", "start": (431, 299), "end": (628, 418)}
        ]
    },
    "CAM_07": {
        # Đã sửa lại tên file (trước đây bị trùng cam06)
        "path": get_path("data/processed_videos/cam07_D2_UngVanKhiem.mp4"), 
        "lines": [
            {"name": "line1", "start": (157, 377), "end": (466, 368)}
        ]
    }
}

# --- CẤU HÌNH MODEL & KAFKA ---
MODEL_PATH = "yolov8s.pt"     # Model YOLO
TARGET_SIZE = (640, 640)      # Resize video để xử lý nhanh
CONF_THRESHOLD = 0.3          # Độ tin cậy tối thiểu
CLASS_IDS = [2, 3, 5, 7]      # Car, Motorcycle, Bus, Truck

# Kafka Config
KAFKA_TOPIC = 'traffic_raw_data'
KAFKA_SERVER = 'localhost:9092'