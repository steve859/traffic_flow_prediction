import cv2

# Đường dẫn video (sửa lại cho đúng với file của bạn)
VIDEO_PATH = "data/processed_videos/resized_7_20_2017 4_59_59 PM (UTC+07_00).mp4"

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"👉 Toạ độ điểm: ({x}, {y})")

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("❌ Không mở được video, kiểm tra lại VIDEO_PATH:", VIDEO_PATH)
    exit(1)

# Đọc đúng 1 frame để hiển thị cố định
ret, frame = cap.read()
if not ret:
    print("❌ Không đọc được frame từ video.")
    cap.release()
    exit(1)

cv2.namedWindow("Chon Vach Dem")
cv2.setMouseCallback("Chon Vach Dem", mouse_callback)

print("ℹ️ HƯỚNG DẪN:")
print("   - Click chuột trái vào các điểm trên frame để lấy toạ độ.")
print("   - Nhấn 'q' để thoát cửa sổ.")

while True:
    cv2.imshow("Chon Vach Dem", frame)
    # 1ms mỗi vòng, đủ để nhận phím 'q' và vẫn giữ frame
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()