import cv2
import os
from tqdm import tqdm

def resize_video(input_path, output_path, target_size=(640, 640)):
    cap = cv2.VideoCapture(input_path)
    
    # Kiểm tra xem có mở được file không (quan trọng với .asf, .mkv cũ)
    if not cap.isOpened():
        print(f"❌ LỖI: Không thể đọc file {input_path}. Bỏ qua!")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: # Phòng trường hợp không đọc được FPS
        fps = 15 
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # --- THAY ĐỔI QUAN TRỌNG ---
    # Luôn sử dụng container .mp4 cho output để tương thích tốt nhất
    # Codec 'mp4v' hoặc 'avc1' (H.264) là tốt nhất
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(output_path, fourcc, fps, target_size)
    
    print(f"🎥 Đang xử lý: {os.path.basename(input_path)} -> {os.path.basename(output_path)}")
    
    for _ in tqdm(range(total_frames), desc="Tiến độ", unit="frame"):
        ret, frame = cap.read()
        if not ret:
            break
            
        try:
            # Resize
            resized_frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
            out.write(resized_frame)
        except Exception as e:
            print(f"Lỗi frame: {e}")
            continue
        
    cap.release()
    out.release()

if __name__ == "__main__":
    os.makedirs("data/processed_videos", exist_ok=True)
    
    # Danh sách file hỗn hợp của bạn
    list_videos = [
        # "7_20_2017 4_59_59 PM (UTC+07_00).mkv", 
        # "7_20_2017 11_59_59 AM (UTC+07_00).mkv",
        # "CongHoa-TruongChinh 2017-07-18_17_00_00_000.asf", 
        # "CongHoa-UtTich1 2017-07-17_14.15.asf",
        # "DuongBaTrac-TaQuangBuu1 2017-07-18_08_00_00_000.asf",
        # "HAMQ1 - 2017-07-20 15-00-07-155.mov", 
        # "NKKN-VoThiSau 2017-07-18_08_00_00_000.asf"
        "CongHoa-UtTich1 2017-07-17_17.15.asf"
    ]
    
    for vid in list_videos:
        in_path = os.path.join("data/raw_videos", vid)
        
        # --- XỬ LÝ TÊN FILE OUTPUT ---
        # 1. Tách tên file và đuôi mở rộng cũ (ví dụ: .asf)
        filename_only, extension = os.path.splitext(vid)
        
        # 2. Tạo tên mới luôn có đuôi .mp4
        new_filename = f"resized_{filename_only}.mp4"
        
        out_path = os.path.join("data/processed_videos", new_filename)
        
        # Kiểm tra file input có tồn tại không trước khi chạy
        if os.path.exists(in_path):
            resize_video(in_path, out_path, target_size=(640, 640))
        else:
            print(f"⚠️ Không tìm thấy file: {in_path}")