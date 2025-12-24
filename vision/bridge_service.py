# src/vision/bridge_service.py
from fastapi import FastAPI, Request
from kafka import KafkaProducer
import json
import uvicorn

app = FastAPI()

# Cấu hình kết nối Kafka (Chạy ở Local nên dùng localhost)
try:
    producer = KafkaProducer(
        bootstrap_servers=['localhost:9092'],
        value_serializer=lambda x: json.dumps(x).encode('utf-8')
    )
    print("✅ Đã kết nối Kafka thành công!")
except Exception as e:
    print(f"❌ Lỗi kết nối Kafka: {e}")
    producer = None

@app.post("/ingest")
async def ingest_data(request: Request):
    try:
        data = await request.json()
        
        # In ra để bạn thấy dữ liệu đang bay về
        print(f"📥 Nhận từ Colab: {data.get('vehicle_type')} - ID: {data.get('vehicle_id')}")
        
        # Đẩy vào Kafka Topic 'traffic_raw_data'
        if producer:
            producer.send('traffic_raw_data', value=data)
            
        return {"status": "ok", "message": "Data pushed to Kafka"}
    except Exception as e:
        print(f"❌ Lỗi xử lý dữ liệu: {e}")
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    # Chạy server tại port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)