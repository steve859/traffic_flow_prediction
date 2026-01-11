"""
slot_fetch.py
Fetch traffic camera images every run (scheduled by Windows Task Scheduler)

- Fetch 1 image per camera
- Save with timestamp
- Designed for 5-min or 15-min interval crawling
"""

import requests
import os
from datetime import datetime
from pathlib import Path
import time

# =========================
# CONFIG
# =========================

# Thư mục lưu ảnh
BASE_DIR = Path(r"D:\Code\traffic_flow_prediction\data\raw_images")

# Danh sách camera (ví dụ)
CAMERAS = {
    "cam1": "https://www.giaothong.hochiminhcity.gov.vn/expandcameraplayer/?camId=5d8cdbdc766c88001718896a&camLocation=Ho%C3%A0ng%20V%C4%83n%20Th%E1%BB%A5%20-%20C%E1%BB%99ng%20H%C3%B2a&camMode=camera&videoUrl=https://d2zihajmogu5jn.cloudfront.net/bipbop-advanced/bipbop_16x9_variant.m3u8",
    "cam2": "https://www.giaothong.hochiminhcity.gov.vn/expandcameraplayer/?camId=587476e3b807da0011e33cee&camLocation=C%E1%BB%99ng%20H%C3%B2a%20-%20%C3%9At%20T%E1%BB%8Bch%201&camMode=camera&videoUrl=https://d2zihajmogu5jn.cloudfront.net/bipbop-advanced/bipbop_16x9_variant.m3u8",
    "cam3": "https://www.giaothong.hochiminhcity.gov.vn/expandcameraplayer/?camId=58ad6214bd82540010390be2&camLocation=C%E1%BB%99ng%20H%C3%B2a%20-%20Ho%C3%A0ng%20Hoa%20Th%C3%A1m&camMode=camera&videoUrl=https://d2zihajmogu5jn.cloudfront.net/bipbop-advanced/bipbop_16x9_variant.m3u8",
    "cam4": "https://www.giaothong.hochiminhcity.gov.vn/expandcameraplayer/?camId=662b57471afb9c00172d9095&camLocation=C%E1%BB%99ng%20H%C3%B2a%20-%20%E1%BA%A4p%20B%E1%BA%AFc&camMode=camera&videoUrl=https://d2zihajmogu5jn.cloudfront.net/bipbop-advanced/bipbop_16x9_variant.m3u8",
    "cam5": "https://www.giaothong.hochiminhcity.gov.vn/expandcameraplayer/?camId=586e1f18f9fab7001111b0a5&camLocation=C%E1%BB%99ng%20H%C3%B2a%20-%20Tr%C6%B0%E1%BB%9Dng%20Chinh&camMode=camera&videoUrl=https://d2zihajmogu5jn.cloudfront.net/bipbop-advanced/bipbop_16x9_variant.m3u8",
    "cam6": "https://www.giaothong.hochiminhcity.gov.vn/expandcameraplayer/?camId=586e25e1f9fab7001111b0ae&camLocation=Tr%C6%B0%E1%BB%9Dng%20Chinh%20-%20T%C3%A2n%20K%E1%BB%B3%20T%C3%A2n%20Qu%C3%BD&camMode=camera&videoUrl=http://camera.thongtingiaothong.vn/s/586e25e1f9fab7001111b0ae/index.m3u8",
    "cam7": "https://www.giaothong.hochiminhcity.gov.vn/expandcameraplayer/?camId=56df807bc062921100c143da&camLocation=Tr%C6%B0%E1%BB%9Dng%20Chinh%20-%20%C3%82u%20C%C6%A1&camMode=camera&videoUrl=https://d2zihajmogu5jn.cloudfront.net/bipbop-advanced/bipbop_16x9_variant.m3u8",
    "cam8": "https://www.giaothong.hochiminhcity.gov.vn/expandcameraplayer/?camId=587478d8b807da0011e33cf3&camLocation=N%C3%BAt%20giao%20B%E1%BA%A3y%20Hi%E1%BB%81n%201%20(C%C3%A1ch%20M%E1%BA%A1ng%20Th%C3%A1ng%20T%C3%A1m)&camMode=camera&videoUrl=https://d2zihajmogu5jn.cloudfront.net/bipbop-advanced/bipbop_16x9_variant.m3u8",
}

# Timeout HTTP
REQUEST_TIMEOUT = 10  # seconds

# =========================
# UTILS
# =========================

def fetch_image(url: str) -> bytes | None:
    try:
        r = requests.get(url, timeout=REQUEST_TIMEOUT)
        if r.status_code == 200 and r.content:
            return r.content
    except Exception as e:
        print(f"[ERROR] Fetch failed: {url} | {e}")
    return None


def save_image(cam_id: str, content: bytes, ts: datetime):
    cam_dir = BASE_DIR / cam_id
    cam_dir.mkdir(parents=True, exist_ok=True)

    fname = ts.strftime("%Y%m%d_%H%M%S") + ".jpg"
    out_path = cam_dir / fname

    with open(out_path, "wb") as f:
        f.write(content)

    print(f"[OK] Saved {cam_id} → {out_path.name}")


# =========================
# MAIN
# =========================

def main():
    print("🚦 Traffic slot fetch started")
    ts = datetime.now()

    for cam_id, url in CAMERAS.items():
        print(f"📡 Fetching {cam_id} ...")
        img = fetch_image(url)
        if img is None:
            print(f"[WARN] No image for {cam_id}")
            continue

        save_image(cam_id, img, ts)
        time.sleep(0.3)  # tránh spam server

    print("✅ Done")


if __name__ == "__main__":
    main()
