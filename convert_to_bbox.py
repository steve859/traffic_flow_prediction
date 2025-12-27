import os

# ===== CONFIG =====
SEG_LABEL_DIR = r"D:\Code\traffic_flow_prediction\labeled_data_to_finetune\detect.v1i.yolov8\train\labels"
DET_LABEL_DIR = r"D:\Code\traffic_flow_prediction\labeled_data_to_finetune\detect.v1i.yolov8\train\labels_det"

PAD = 0.08  # 8% padding – khuyên dùng cho xe máy VN
# ==================

os.makedirs(DET_LABEL_DIR, exist_ok=True)


def seg_line_to_bbox(line, pad=0.08):
    nums = list(map(float, line.strip().split()))
    cls = int(nums[0])
    coords = nums[1:]

    xs = coords[0::2]
    ys = coords[1::2]

    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)

    # padding
    w = xmax - xmin
    h = ymax - ymin
    xmin = max(0.0, xmin - w * pad)
    ymin = max(0.0, ymin - h * pad)
    xmax = min(1.0, xmax + w * pad)
    ymax = min(1.0, ymax + h * pad)

    x_center = (xmin + xmax) / 2
    y_center = (ymin + ymax) / 2
    width = xmax - xmin
    height = ymax - ymin

    return f"{cls} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"


def convert_file(seg_path, out_path):
    with open(seg_path, "r") as f:
        lines = f.readlines()

    out_lines = []
    for line in lines:
        if line.strip():
            out_lines.append(seg_line_to_bbox(line, PAD))

    with open(out_path, "w") as f:
        f.writelines(out_lines)


def main():
    for fname in os.listdir(SEG_LABEL_DIR):
        if fname.endswith(".txt"):
            seg_path = os.path.join(SEG_LABEL_DIR, fname)
            out_path = os.path.join(DET_LABEL_DIR, fname)
            convert_file(seg_path, out_path)

    print(f"[DONE] Converted labels saved to: {DET_LABEL_DIR}")


if __name__ == "__main__":
    main()
