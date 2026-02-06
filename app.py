"""
🎯 HUMAN DETECTION AI
Ứng dụng nhận dạng người vs không phải người
Sinh viên: Lê Quang Đạo | MSSV: 223332821
"""

import streamlit as st
from tensorflow import keras
from PIL import Image
import numpy as np
import requests
from io import BytesIO

# ==================== CONFIG ====================
st.set_page_config(
    page_title="Human Detection AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==================== CUSTOM CSS ====================
st.markdown("""
<style>
    * {
        margin: 0;
        padding: 0;
    }
    
    body {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
        padding: 0;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        color: white;
    }
    
    .stTabs [aria-selected="true"] {
        background: white !important;
        color: #667eea !important;
        font-weight: bold;
    }
    
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 50px 20px;
        border-radius: 0;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    .header-container h1 {
        font-size: 3em;
        margin-bottom: 10px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .header-container p {
        font-size: 1.2em;
        opacity: 0.9;
    }
    
    .result-container {
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        margin-top: 30px;
        font-weight: bold;
        font-size: 1.5em;
        box-shadow: 0 8px 20px rgba(0,0,0,0.2);
        animation: slideUp 0.5s ease;
    }
    
    @keyframes slideUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .result-human {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
    }
    
    .result-non-human {
        background: linear-gradient(135deg, #ee0979 0%, #ff6a00 100%);
        color: white;
    }
    
    .confidence-meter {
        margin-top: 20px;
        height: 10px;
        background: rgba(255,255,255,0.3);
        border-radius: 5px;
        overflow: hidden;
    }
    
    .confidence-fill {
        height: 100%;
        background: white;
        border-radius: 5px;
        transition: width 0.3s ease;
    }
    
    .info-box {
        background: white;
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    }
    
    .student-info {
        background: rgba(255, 255, 255, 0.95);
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 4px solid #667eea;
    }
    
    .input-section {
        background: white;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    
    .footer {
        text-align: center;
        padding: 20px;
        color: white;
        margin-top: 50px;
        font-size: 12px;
        opacity: 0.8;
    }
</style>
""", unsafe_allow_html=True)

# ==================== HEADER ====================
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("""
    <div class="header-container">
        <h1>🤖 HUMAN DETECTION AI</h1>
        <p>Công nghệ Deep Learning nhận dạng người vs không phải người</p>
    </div>
    """, unsafe_allow_html=True)

# Student Info
st.markdown("""
<div class="student-info">
    <b>👤 Tác giả:</b> Lê Quang Đạo | <b>🎓 MSSV:</b> 223332821
</div>
""", unsafe_allow_html=True)

# ==================== CONSTANTS ====================
IMG_SIZE = 64

# ==================== FUNCTIONS ====================
@st.cache_resource
def load_model():
    """Load pre-trained model"""
    try:
        model = keras.models.load_model('humantachi.h5')
        return model
    except Exception as e:
        return None

def preprocess_image(image):
    """Preprocess image for prediction"""
    image = image.resize((IMG_SIZE, IMG_SIZE))
    if image.mode != 'RGB':
        image = image.convert('RGB')
    img_array = np.array(image, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict(model, image):
    """Predict image label"""
    img_array = preprocess_image(image)
    prediction = model.predict(img_array, verbose=0)[0][0]
    return prediction

def show_result(prediction):
    """Display prediction result with styling"""
    if prediction > 0.5:
        confidence = prediction * 100
        label = "KHÔNG PHẢI NGƯỜI"
        emoji = "❌"
        class_name = "result-non-human"
    else:
        confidence = (1 - prediction) * 100
        label = "LÀ NGƯỜI"
        emoji = "✅"
        class_name = "result-human"
    
    st.markdown(f"""
    <div class="result-container {class_name}">
        {emoji} <br> <br>
        {label}
        <div class="confidence-meter">
            <div class="confidence-fill" style="width: {confidence}%"></div>
        </div>
        <div style="margin-top: 15px; font-size: 1.1em;">
            Độ tin cậy: <b>{confidence:.1f}%</b>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ==================== MAIN APP ====================
model = load_model()

if model is not None:
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📤 Tải Ảnh Lên",
        "📷 Webcam",
        "🔗 Link Ảnh",
        "ℹ️ Hướng Dẫn"
    ])
    
    # ===== TAB 1: Upload File =====
    with tab1:
        st.markdown("""
        <div class="input-section">
            <h3>📤 Tải ảnh từ máy tính của bạn</h3>
            <p>Chọn một ảnh JPG, PNG, BMP hoặc WEBP</p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Chọn ảnh",
            type=['jpg', 'jpeg', 'png', 'bmp', 'webp'],
            key="upload_file"
        )
        
        if uploaded_file:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                image = Image.open(uploaded_file)
                st.image(image, caption="Ảnh của bạn", use_container_width=True)
            
            with col2:
                st.markdown("### 🔍 Kết Quả Phân Tích")
                if st.button("Phân Tích Ảnh", type="primary", use_container_width=True, key="btn_file"):
                    with st.spinner("⏳ Đang xử lý..."):
                        prediction = predict(model, image)
                        show_result(prediction)
    
    # ===== TAB 2: Webcam =====
    with tab2:
        st.markdown("""
        <div class="input-section">
            <h3>📷 Chụp ảnh từ webcam</h3>
            <p>Cho phép truy cập webcam để chụp ảnh</p>
        </div>
        """, unsafe_allow_html=True)
        
        picture = st.camera_input("Chụp ảnh")
        
        if picture:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                image = Image.open(picture)
                st.image(image, caption="Ảnh chụp", use_container_width=True)
            
            with col2:
                st.markdown("### 🔍 Kết Quả Phân Tích")
                if st.button("Phân Tích Ảnh", type="primary", use_container_width=True, key="btn_camera"):
                    with st.spinner("⏳ Đang xử lý..."):
                        prediction = predict(model, image)
                        show_result(prediction)
    
    # ===== TAB 3: Image URL =====
    with tab3:
        st.markdown("""
        <div class="input-section">
            <h3>🔗 Phân tích ảnh từ link</h3>
            <p>Dán link ảnh từ internet (https://...)</p>
        </div>
        """, unsafe_allow_html=True)
        
        url = st.text_input(
            "Link ảnh",
            placeholder="https://example.com/image.jpg",
            key="image_url"
        )
        
        if url:
            if st.button("Tải & Phân Tích", type="primary", use_container_width=True):
                try:
                    with st.spinner("⏳ Đang tải ảnh..."):
                        response = requests.get(url, timeout=10)
                        response.raise_for_status()
                        image = Image.open(BytesIO(response.content))
                        
                        col1, col2 = st.columns([1, 1])
                        
                        with col1:
                            st.image(image, caption="Ảnh từ link", use_container_width=True)
                        
                        with col2:
                            st.markdown("### 🔍 Kết Quả Phân Tích")
                            with st.spinner("⏳ Đang xử lý..."):
                                prediction = predict(model, image)
                                show_result(prediction)
                
                except requests.exceptions.MissingSchema:
                    st.error("❌ Link không hợp lệ. Sử dụng http:// hoặc https://")
                except requests.exceptions.ConnectionError:
                    st.error("❌ Không thể kết nối. Kiểm tra link hoặc internet.")
                except requests.exceptions.Timeout:
                    st.error("❌ Hết thời gian chờ. Link có thể không hoạt động.")
                except Exception as e:
                    st.error(f"❌ Lỗi: {str(e)}")
    
    # ===== TAB 4: Guide =====
    with tab4:
        st.markdown("""
        <div class="info-box">
            <h3>📖 Hướng Dẫn Sử Dụng</h3>
            
            <h4>✨ Tính Năng:</h4>
            <ul>
                <li><b>📤 Upload Ảnh:</b> Tải ảnh từ máy tính</li>
                <li><b>📷 Webcam:</b> Chụp ảnh trực tiếp</li>
                <li><b>🔗 URL:</b> Sử dụng link ảnh online</li>
            </ul>
            
            <h4>📋 Định Dạng Hỗ Trợ:</h4>
            <p>JPG, JPEG, PNG, BMP, WEBP</p>
            
            <h4>🎯 Cách Sử Dụng:</h4>
            <ol>
                <li>Chọn tab phù hợp (Upload, Webcam, hoặc URL)</li>
                <li>Cung cấp ảnh input</li>
                <li>Nhấn nút "Phân Tích Ảnh"</li>
                <li>Xem kết quả và độ tin cậy</li>
            </ol>
            
            <h4>⚙️ Mô Hình:</h4>
            <p><b>Architecture:</b> CNN (Convolutional Neural Network)</p>
            <p><b>Input Size:</b> 64x64 pixels</p>
            <p><b>Classes:</b> 2 (Người / Không phải người)</p>
            <p><b>Model File:</b> humantachi.h5</p>
            
            <h4>💡 Lưu Ý:</h4>
            <ul>
                <li>Ảnh càng rõ ràng, kết quả càng chính xác</li>
                <li>Tránh ảnh quá nhỏ hoặc quá mờ</li>
                <li>Độ tin cậy trên 50% = Không phải người</li>
                <li>Độ tin cậy dưới 50% = Là người</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

else:
    st.error("❌ Không tìm thấy model (humantachi.h5)")
    st.info("""
    **Giải pháp:**
    1. Đặt file `humantachi.h5` cùng thư mục với `app.py`
    2. Hoặc chạy script huấn luyện trước
    3. Kiểm tra tên file model
    """)

# ==================== FOOTER ====================
st.markdown("""
<div class="footer">
    <p>🎓 Deep Learning Project - Human Detection System</p>
    <p>© 2026 Lê Quang Đạo | Powered by TensorFlow & Streamlit</p>
</div>
""", unsafe_allow_html=True)

