import cv2
from ultralytics import YOLO
import supervision as sv
import numpy as np

# --- CẤU HÌNH ---
VIDEO_PATH = "data/processed_videos/resized_7_20_2017 4_59_59 PM.mp4"
MODEL_PATH = "yolov8n.pt" # Dùng bản nano cho nhanh

# Toạ độ vạch đếm (Lấy từ Bước 1) - Ví dụ: Điểm đầu (100, 400), Điểm cuối (600, 400)
# Bạn thay số này bằng toạ độ thực tế bạn vừa tìm được
LINE_START = sv.Point(50, 400) 
LINE_END = sv.Point(600, 400)

def main():
    # 1. Load Model & Video
    model = YOLO(MODEL_PATH)
    
    # Lấy thông tin video để lưu video kết quả (nếu muốn)
    video_info = sv.VideoInfo.from_video_path(VIDEO_PATH)
    
    # 2. Setup ByteTrack (Tracker)
    # ByteTrack giúp nhớ ID của xe qua các khung hình
    tracker = sv.ByteTrack(track_thresh=0.25, track_buffer=30, match_thresh=0.8, frame_rate=30)
    
    # 3. Setup LineZone (Vùng đếm)
    line_zone = sv.LineZone(start=LINE_START, end=LINE_END)
    
    # Setup Annotators (Để vẽ lên hình cho đẹp)
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    trace_annotator = sv.TraceAnnotator() # Vẽ đuôi di chuyển của xe
    line_zone_annotator = sv.LineZoneAnnotator(thickness=2, text_thickness=1, text_scale=0.5)

    # 4. Process Video Loop
    # Dùng generator của supervision để đọc frame cho tiện
    frame_generator = sv.get_video_frames_generator(VIDEO_PATH)

    print("🚀 Bắt đầu xử lý...")
    
    for frame in frame_generator:
        # a. Detect bằng YOLO
        results = model(frame, verbose=False)[0]
        
        # b. Convert kết quả sang format của Supervision
        detections = sv.Detections.from_ultralytics(results)
        
        # Chỉ lấy các class xe cộ (Car, motorcycle, bus, truck)
        # COCO IDs: 2=car, 3=motorcycle, 5=bus, 7=truck
        detections = detections[np.isin(detections.class_id, [2, 3, 5, 7])]

        # c. Update Tracker (Gán ID cho xe)
        detections = tracker.update_with_detections(detections)
        
        # d. Kiểm tra vượt vạch (Line Crossing) -> QUAN TRỌNG NHẤT
        cross_in, cross_out = line_zone.trigger(detections)
        
        # e. In ra console nếu có xe qua vạch (Giả lập gửi Kafka tại đây)
        if np.any(cross_in) or np.any(cross_out):
            print(f"📈 Xe vào: {line_zone.in_count} | Xe ra: {line_zone.out_count}")

        # f. Vẽ lên hình để debug (Optional)
        labels = [
            f"#{tracker_id} {model.model.names[class_id]}"
            for tracker_id, class_id
            in zip(detections.tracker_id, detections.class_id)
        ]
        
        annotated_frame = trace_annotator.annotate(scene=frame.copy(), detections=detections)
        annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
        line_zone_annotator.annotate(annotated_frame, line_counter=line_zone)

        cv2.imshow("Traffic Counting Debug", annotated_frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()