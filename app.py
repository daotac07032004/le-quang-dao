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
        box-sizing: border-box;
    }
    
    html, body, [data-testid="stAppViewContainer"] {
        background: #0f172a;
        color: #e2e8f0;
    }
    
    .main {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        border-bottom: 2px solid #334155;
        gap: 0;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 12px 24px;
        background: transparent;
        color: #94a3b8;
        border-radius: 0;
        font-weight: 500;
        border-bottom: 3px solid transparent;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        color: #64748b;
    }
    
    .stTabs [aria-selected="true"] {
        background: transparent !important;
        color: #60a5fa !important;
        border-bottom: 3px solid #60a5fa !important;
    }
    
    .card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 24px;
        margin: 16px 0;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
    }
    
    .card:hover {
        border-color: #60a5fa;
        box-shadow: 0 10px 40px rgba(96, 165, 250, 0.1);
    }
    
    .header-title {
        font-size: 2.5em;
        font-weight: 700;
        background: linear-gradient(90deg, #60a5fa 0%, #a78bfa 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin: 30px 0 10px 0;
        letter-spacing: -1px;
    }
    
    .header-subtitle {
        text-align: center;
        color: #94a3b8;
        font-size: 1.1em;
        margin-bottom: 30px;
    }
    
    .student-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border-left: 4px solid #60a5fa;
        border: 1px solid #334155;
        border-left: 4px solid #60a5fa;
        padding: 16px;
        border-radius: 8px;
        margin: 20px 0;
        text-align: center;
    }
    
    .student-card p {
        margin: 8px 0;
        color: #cbd5e1;
    }
    
    .student-card b {
        color: #60a5fa;
    }
    
    .result-box {
        padding: 40px 30px;
        border-radius: 12px;
        text-align: center;
        margin: 20px 0;
        font-weight: 600;
        font-size: 1.3em;
        border: 2px solid transparent;
        animation: slideIn 0.5s ease;
    }
    
    @keyframes slideIn {
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
        background: linear-gradient(135deg, #10b981 15%, #059669 100%);
        border-color: #10b981;
        color: white;
        box-shadow: 0 0 30px rgba(16, 185, 129, 0.2);
    }
    
    .result-non-human {
        background: linear-gradient(135deg, #ef4444 15%, #dc2626 100%);
        border-color: #ef4444;
        color: white;
        box-shadow: 0 0 30px rgba(239, 68, 68, 0.2);
    }
    
    .confidence-bar {
        width: 100%;
        height: 12px;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 6px;
        margin: 20px 0;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .confidence-fill {
        height: 100%;
        background: linear-gradient(90deg, #60a5fa, #a78bfa);
        border-radius: 6px;
        transition: width 0.5s ease;
    }
    
    .section-title {
        font-size: 1.5em;
        font-weight: 600;
        color: #60a5fa;
        margin: 20px 0 15px 0;
        border-left: 3px solid #60a5fa;
        padding-left: 12px;
    }
    
    .info-text {
        color: #cbd5e1;
        line-height: 1.6;
    }
    
    .footer {
        text-align: center;
        padding: 30px 20px;
        color: #64748b;
        margin-top: 30px;
        border-top: 1px solid #334155;
        font-size: 14px;
    }
    
    .input-label {
        color: #94a3b8;
        font-size: 0.95em;
        margin-bottom: 8px;
    }
</style>
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
        label = "🚫 KHÔNG PHẢI NGƯỜI"
        emoji = ""
        class_name = "result-non-human"
    else:
        confidence = (1 - prediction) * 100
        label = "✅ LÀ NGƯỜI"
        emoji = ""
        class_name = "result-human"
    
    st.markdown(f"""
    <div class="result-box {class_name}">
        {label}
        <div class="confidence-bar">
            <div class="confidence-fill" style="width: {confidence}%"></div>
        </div>
        <div style="margin-top: 15px; font-size: 1em;">
            Độ tin cậy: <b>{confidence:.1f}%</b>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ==================== MAIN APP ====================
st.markdown('<div class="header-title">🤖 HUMAN DETECTION</div>', unsafe_allow_html=True)
st.markdown('<div class="header-subtitle">AI-Powered Person Recognition System</div>', unsafe_allow_html=True)

st.markdown("""
<div class="student-card">
    <p><b>👨‍💼 Developer:</b> Lê Quang Đạo</p>
    <p><b>🎓 Student ID:</b> 223332821</p>
</div>
""", unsafe_allow_html=True)

st.divider()

model = load_model()

if model is not None:
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📁 Upload Image",
        "📸 Webcam",
        "🌐 Image URL",
        "📚 Guide"
    ])
    
    # ===== TAB 1: Upload File =====
    with tab1:
        st.markdown('<div class="section-title">📁 Select Image from Device</div>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['jpg', 'jpeg', 'png', 'bmp', 'webp'],
            key="upload_file"
        )
        
        if uploaded_file:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                image = Image.open(uploaded_file)
                st.image(image, caption="Your Image", use_container_width=True)
            
            with col2:
                st.markdown('<div class="section-title">Analysis Result</div>', unsafe_allow_html=True)
                if st.button("🔍 Analyze", type="primary", use_container_width=True, key="btn_file"):
                    with st.spinner("Processing image..."):
                        prediction = predict(model, image)
                        show_result(prediction)
    
    # ===== TAB 2: Webcam =====
    with tab2:
        st.markdown('<div class="section-title">📸 Capture from Webcam</div>', unsafe_allow_html=True)
        
        picture = st.camera_input("Take a photo")
        
        if picture:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                image = Image.open(picture)
                st.image(image, caption="Captured Image", use_container_width=True)
            
            with col2:
                st.markdown('<div class="section-title">Analysis Result</div>', unsafe_allow_html=True)
                if st.button("🔍 Analyze", type="primary", use_container_width=True, key="btn_camera"):
                    with st.spinner("Processing image..."):
                        prediction = predict(model, image)
                        show_result(prediction)
    
    # ===== TAB 3: Image URL =====
    with tab3:
        st.markdown('<div class="section-title">🌐 Analyze from URL</div>', unsafe_allow_html=True)
        
        url = st.text_input(
            "Image URL",
            placeholder="https://example.com/image.jpg",
            key="image_url"
        )
        
        if url:
            if st.button("Load & Analyze", type="primary", use_container_width=True):
                try:
                    with st.spinner("Loading image..."):
                        response = requests.get(url, timeout=10)
                        response.raise_for_status()
                        image = Image.open(BytesIO(response.content))
                        
                        col1, col2 = st.columns([1, 1])
                        
                        with col1:
                            st.image(image, caption="Image from URL", use_container_width=True)
                        
                        with col2:
                            st.markdown('<div class="section-title">Analysis Result</div>', unsafe_allow_html=True)
                            with st.spinner("Processing image..."):
                                prediction = predict(model, image)
                                show_result(prediction)
                
                except requests.exceptions.MissingSchema:
                    st.error("❌ Invalid URL. Use http:// or https://")
                except requests.exceptions.ConnectionError:
                    st.error("❌ Cannot connect. Check URL or internet.")
                except requests.exceptions.Timeout:
                    st.error("❌ Timeout. URL may be unavailable.")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    
    # ===== TAB 4: Guide =====
    with tab4:
        st.markdown("""
        <div class="card">
            <div class="section-title">📖 User Guide</div>
            
            <p style="color: #cbd5e1; margin: 15px 0;"><b style="color: #60a5fa;">✨ Features:</b></p>
            <ul style="color: #cbd5e1; line-height: 1.8;">
                <li>📁 <b>Upload:</b> Upload image from your device</li>
                <li>📸 <b>Webcam:</b> Capture image directly</li>
                <li>🌐 <b>URL:</b> Use online image link</li>
            </ul>
            
            <p style="color: #cbd5e1; margin: 15px 0;"><b style="color: #60a5fa;">📋 Supported Formats:</b></p>
            <p style="color: #cbd5e1;">JPG, JPEG, PNG, BMP, WEBP</p>
            
            <p style="color: #cbd5e1; margin: 15px 0;"><b style="color: #60a5fa;">🎯 How to Use:</b></p>
            <ol style="color: #cbd5e1; line-height: 1.8;">
                <li>Choose a tab (Upload, Webcam, or URL)</li>
                <li>Provide image input</li>
                <li>Click "Analyze" button</li>
                <li>View results and confidence</li>
            </ol>
            
            <p style="color: #cbd5e1; margin: 15px 0;"><b style="color: #60a5fa;">⚙️ Model Details:</b></p>
            <ul style="color: #cbd5e1; line-height: 1.8;">
                <li><b>Architecture:</b> CNN (Convolutional Neural Network)</li>
                <li><b>Input Size:</b> 64x64 pixels</li>
                <li><b>Classes:</b> Person / Non-Person</li>
                <li><b>Model:</b> humantachi.h5</li>
            </ul>
            
            <p style="color: #cbd5e1; margin: 15px 0;"><b style="color: #60a5fa;">💡 Tips:</b></p>
            <ul style="color: #cbd5e1; line-height: 1.8;">
                <li>Clear images produce better results</li>
                <li>Avoid very small or blurry images</li>
                <li>Confidence > 50% = Non-Person</li>
                <li>Confidence < 50% = Person</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

else:
    st.error("❌ Model not found (humantachi.h5)")
    st.info("""
    **Solution:**
    1. Place `humantachi.h5` in the same directory as `app.py`
    2. Or run the training script first
    3. Check the model filename
    """)

# ==================== FOOTER ====================
st.markdown("""
<div class="footer">
    <p>🎓 Deep Learning Project - Human Detection System</p>
    <p>© 2026 Lê Quang Đạo | Powered by TensorFlow & Streamlit</p>
</div>
""", unsafe_allow_html=True)

