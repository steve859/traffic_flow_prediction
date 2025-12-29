import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time
import numpy as np

# Cấu hình trang
st.set_page_config(page_title="HCMC Traffic Dashboard", layout="wide")

st.title("🚦 Dashboard Phân Tích & Dự Báo Giao Thông TP.HCM")
st.markdown("**Module:** Integration, Dashboard & Report")

# Sidebar: Nhập dữ liệu
st.sidebar.header("Dữ Liệu Đầu Vào")
uploaded_flow = st.sidebar.file_uploader("Tải file hcmc_flow.csv", type="csv")
uploaded_pred = st.sidebar.file_uploader("Tải file prediction.csv", type="csv")

# --- 1. Xử lý dữ liệu ---
if uploaded_flow and uploaded_pred:
    flow_df = pd.read_csv(uploaded_flow)
    pred_df = pd.read_csv(uploaded_pred)
    
    # Tính tổng lưu lượng
    flow_df['total_flow'] = flow_df[['motorbike', 'car', 'bus', 'truck']].sum(axis=1)
    
    st.sidebar.success("Đã tải dữ liệu thành công!")

    # --- 2. Đánh giá Mô hình ---
    st.header("1. Hiệu Suất Mô Hình (Thực tế vs Dự báo)")
    
    col1, col2, col3 = st.columns(3)
    
    # Tính chỉ số
    mae = np.mean(np.abs(pred_df['y_true'] - pred_df['y_pred']))
    correlation = pred_df['y_true'].corr(pred_df['y_pred'])
    
    col1.metric("MAE (Sai số tuyệt đối)", f"{mae:.2f}")
    col2.metric("Độ tương quan (Correlation)", f"{correlation:.2f}")
    col3.metric("Số lượng mẫu", f"{len(pred_df)}")

    # Biểu đồ đường so sánh
    fig_line = go.Figure()
    fig_line.add_trace(go.Scatter(y=pred_df['y_true'], mode='lines', name='Thực tế (Real)'))
    fig_line.add_trace(go.Scatter(y=pred_df['y_pred'], mode='lines', name='Dự báo (Forecast)', line=dict(dash='dash')))
    fig_line.update_layout(
        title="So sánh Lưu lượng: Thực tế vs Dự báo", 
        xaxis_title="Thời gian (Time Step)", 
        yaxis_title="Số lượng xe"
    )
    st.plotly_chart(fig_line, use_container_width=True)

    # --- 3. Phân tích Lưu lượng ---
    st.header("2. Phân Tích Lưu Lượng Chi Tiết")
    
    tab1, tab2 = st.tabs(["Bản Đồ Nhiệt (Heatmap)", "Lọc Theo Camera"])
    
    with tab1:
        # Heatmap
        heatmap_data = flow_df.pivot_table(index='camera_id', columns='slot_idx', values='total_flow', aggfunc='mean')
        fig_heat = px.imshow(heatmap_data, 
                             labels=dict(x="Khung giờ (Slot)", y="Camera ID", color="Lưu lượng"),
                             title="Mật độ giao thông theo Camera và Thời gian")
        st.plotly_chart(fig_heat, use_container_width=True)
        
    with tab2:
        # Interactive Chart
        selected_cam = st.selectbox("Chọn Camera để xem chi tiết:", flow_df['camera_id'].unique())
        filtered_df = flow_df[flow_df['camera_id'] == selected_cam]
        
        fig_bar = px.bar(filtered_df, x='slot_idx', y=['motorbike', 'car', 'bus', 'truck'], 
                         title=f"Thành phần phương tiện tại {selected_cam}",
                         labels={"value": "Số lượng", "variable": "Loại xe"})
        st.plotly_chart(fig_bar, use_container_width=True)

    # --- 4. Demo Real-time ---
    st.header("3. Demo Giám Sát Real-time & Dự Báo")
    
    col_video, col_stats = st.columns([2, 1])
    
    with col_video:
        # Placeholder cho video
        st.image("traffic_sample.jpg", caption="Camera Feed (Local)", use_container_width=True)
    
    with col_stats:
        st.subheader("Trạng thái hiện tại")
        placeholder = st.empty()
        start_btn = st.button("Bắt đầu mô phỏng")
        
        if start_btn:
            for i in range(10):
                # Giả lập số liệu nhảy
                current_flow = np.random.randint(20, 100)
                pred_next_30 = current_flow * (1 + np.random.uniform(-0.1, 0.1))
                
                with placeholder.container():
                    st.metric("Lưu lượng hiện tại", f"{current_flow} xe/phút", delta=f"{np.random.randint(-5, 5)}")
                    st.metric("Dự báo 30p tới", f"{int(pred_next_30)} xe/phút")
                    st.progress(current_flow % 100)
                
                time.sleep(0.5)

    # --- 5. Xuất Báo Cáo ---
    st.header("4. Xuất Dữ Liệu")
    csv = flow_df.to_csv(index=False).encode('utf-8')
    st.download_button("Tải xuống dữ liệu đã xử lý (CSV)", csv, "processed_traffic_data.csv", "text/csv")

else:
    st.info("Vui lòng tải lên cả 2 file 'hcmc_flow.csv' và 'prediction.csv' ở thanh bên trái để bắt đầu.")