import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# --- 1. THIẾT LẬP GIAO DIỆN (Long bào) ---
st.set_page_config(
    page_title="Human Detection AI",
    page_icon="👑",
    layout="centered"
)

# --- 2. HÀM LOAD MODEL (Triệu hồi thần thú) ---
# Dùng cache để model chỉ cần load 1 lần duy nhất, giúp web chạy nhanh
@st.cache_resource 
def load_model():
    # Đảm bảo tên file này khớp y hệt file Bệ hạ tải từ Colab về
    model_path = 'humantachi.h5'
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except OSError:
        return None

# --- 3. GIAO DIỆN CHÍNH ---
st.title("👤 Hệ Thống Nhận Diện: NGƯỜI hay VẬT?")
st.write("---")
st.info("Bệ hạ hãy ban cho thần một tấm ảnh, thần sẽ soi xét xem đó là Người hay Không phải người.")

# Load model ngay khi vào web
model = load_model()

if model is None:
    st.error("⚠️ LỖI: Không tìm thấy file 'human_detection_model.h5'. Bệ hạ hãy nhớ tải file model lên cùng thư mục với file app.py này nhé!")
else:
    # --- 4. KHU VỰC TẢI ẢNH ---
    uploaded_file = st.file_uploader("Chọn ảnh để tải lên...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Hiển thị ảnh
        image = Image.open(uploaded_file)
        st.image(image, caption='Ảnh đã tải lên', use_container_width=True)
        
        # Nút bấm bắt đầu dự đoán
        if st.button("🔍 Phân tích ngay"):
            with st.spinner('Thần đang tính toán... xin Bệ hạ đợi trong giây lát...'):
                try:
                    # --- 5. TIỀN XỬ LÝ ẢNH (Phải giống hệt lúc Train) ---
                    # Resize về 224x224
                    img = image.resize((224, 224))
                    img_array = np.array(img)

                    # Nếu ảnh có 4 kênh màu (PNG trong suốt), bỏ kênh Alpha đi chỉ lấy RGB
                    if img_array.shape[-1] == 4:
                        img_array = img_array[:, :, :3]
                    
                    # Chuẩn hóa về khoảng [0, 1]
                    img_array = img_array / 255.0
                    
                    # Thêm chiều batch (1, 224, 224, 3)
                    img_array = np.expand_dims(img_array, axis=0)

                    # --- 6. DỰ ĐOÁN ---
                    prediction = model.predict(img_array)[0][0]
                    
                    # Ngưỡng phân loại (Threshold)
                    threshold = 0.5
                    
                    st.divider()
                    
                    # --- 7. HIỂN THỊ KẾT QUẢ ---
                    if prediction > threshold:
                        confidence = prediction * 100
                        st.success(f"🎉 Kết quả: ĐÂY LÀ CON NGƯỜI")
                        st.metric(label="Độ tin cậy", value=f"{confidence:.2f}%")
                        if confidence > 90:
                            st.balloons() # Thả bóng bay chúc mừng
                    else:
                        confidence = (1 - prediction) * 100
                        st.warning(f"🤖 Kết quả: KHÔNG PHẢI NGƯỜI")
                        st.metric(label="Độ tin cậy", value=f"{confidence:.2f}%")
                        
                except Exception as e:
                    st.error(f"Có lỗi xảy ra khi xử lý ảnh: {e}")

# Footer
st.markdown("---")

st.caption("Được phát triển bởi Bệ hạ anh minh.")
