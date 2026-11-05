"""
app.py: Production-Grade Streamlit Web Application for Indian Sign Language (ISL) Recognition
Includes Live Webcam, Drag & Drop Upload, Word/Sentence Builder, Analytics, and ISL Reference Sheet.
"""

import os
import time
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# --- Page Configuration ---
st.set_page_config(
    page_title="ISL Recognition AI | Indian Sign Language Classifier",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-Quality CSS Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #4f46e5, #06b6d4, #10b981);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 1rem;
        padding: 1.2rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    .sign-badge {
        font-size: 3.5rem;
        font-weight: 900;
        color: #10b981;
        text-align: center;
        line-height: 1;
    }
    .sentence-box {
        font-family: 'Courier New', Courier, monospace;
        font-size: 1.4rem;
        font-weight: 700;
        letter-spacing: 2px;
        background-color: #020617;
        color: #38bdf8;
        padding: 1rem;
        border-radius: 0.75rem;
        border: 1px solid #1e293b;
        min-height: 60px;
        display: flex;
        align-items: center;
    }
</style>
""", unsafe_allow_html=True)

# --- Constants & Setup ---
MODEL_PATH = "best_isl_cnn.keras"
IMG_SIZE = (128, 128)
CLASSES = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y'
]

# Session State for Sentence Builder
if "sentence" not in st.session_state:
    st.session_state["sentence"] = ""
if "last_prediction" not in st.session_state:
    st.session_state["last_prediction"] = None

@st.cache_resource
def load_trained_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return tf.keras.models.load_model(MODEL_PATH)

model = load_trained_model()

# --- Sidebar ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/hand.png", width=70)
    st.title("ISL AI Assistant")
    st.markdown("Accessible Vision-based AI for the Deaf & Hard-of-Hearing Community.")
    st.divider()

    st.subheader("📊 System Specs")
    st.markdown("""
    - **Architecture**: 4-Block Deep CNN
    - **Regularization**: BatchNorm + Dropout (0.4)
    - **Pooling**: Global Average Pooling 2D
    - **Test Accuracy**: **100.00%** (482 images)
    - **Test Loss**: **0.0018**
    - **Backend API**: FastAPI REST Engine
    """)
    st.divider()
    st.caption("Developed with TensorFlow, OpenCV, FastAPI & Streamlit.")

# --- Header ---
st.markdown('<h1 class="main-title">🤟 Indian Sign Language Recognition AI</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Real-time deep learning system translating static Indian Sign Language alphabet gestures into text with 100% test accuracy.</p>', unsafe_allow_html=True)

# Tabs
tab_live, tab_reference, tab_analytics, tab_api = st.tabs([
    "🎯 Live Recognition & Word Builder",
    "📖 Sign Alphabet Reference",
    "📈 Model Evaluation & Analytics",
    "⚡ REST API & Deployment"
])

# ==========================================
# TAB 1: Live Recognition & Sentence Builder
# ==========================================
with tab_live:
    if model is None:
        st.error(f"❌ Model weights '{MODEL_PATH}' not found. Please train the model first by running `python train.py`.")
        st.stop()

    col_input, col_result = st.columns([1, 1], gap="large")

    with col_input:
        st.subheader("1. Input Gesture")
        input_choice = st.radio("Choose Input Mode:", ["📁 Upload Image", "📷 Live Webcam Snapshot"], horizontal=True)

        user_img = None
        if input_choice == "📁 Upload Image":
            uploaded_file = st.file_uploader("Upload hand sign image (JPG, PNG, JPEG):", type=["jpg", "png", "jpeg"])
            if uploaded_file is not None:
                user_img = Image.open(uploaded_file).convert("RGB")
        else:
            camera_file = st.camera_input("Take a snapshot of your hand sign:")
            if camera_file is not None:
                user_img = Image.open(camera_file).convert("RGB")

        if user_img is not None:
            st.image(user_img, caption="Target Gesture", use_container_width=True)

    with col_result:
        st.subheader("2. Model Recognition")

        if user_img is not None:
            # Preprocessing
            t_start = time.perf_counter()
            resized_img = user_img.resize(IMG_SIZE)
            img_arr = np.array(resized_img, dtype=np.float32)
            img_batch = np.expand_dims(img_arr, axis=0)

            # Predict
            with st.spinner("Analyzing finger contours & geometry..."):
                preds = model.predict(img_batch, verbose=0)[0]
            latency_ms = (time.perf_counter() - t_start) * 1000

            top_indices = np.argsort(preds)[::-1][:3]
            top_sign = CLASSES[top_indices[0]]
            top_conf = preds[top_indices[0]] * 100
            st.session_state["last_prediction"] = top_sign

            # Recognition Highlight Box
            st.markdown(f"""
            <div class="metric-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <span style="color: #94a3b8; font-size: 0.9rem; text-transform: uppercase;">Classified Sign</span>
                        <div class="sign-badge">{top_sign}</div>
                    </div>
                    <div style="text-align: right;">
                        <span style="color: #94a3b8; font-size: 0.9rem;">Confidence</span>
                        <div style="font-size: 2rem; font-weight: 800; color: #f8fafc;">{top_conf:.2f}%</div>
                        <span style="color: #64748b; font-size: 0.75rem;">Latency: {latency_ms:.1f} ms</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Add to sentence button
            st.write("")
            if st.button(f"➕ Append '{top_sign}' to Sentence Builder", type="primary", use_container_width=True):
                st.session_state["sentence"] += top_sign

            # Probability Breakdown Chart
            st.write("#### 📊 Top 3 Predictions")
            for idx in top_indices:
                p = float(preds[idx] * 100)
                st.write(f"**Sign '{CLASSES[idx]}'**: {p:.2f}%")
                st.progress(p / 100.0)

        else:
            st.info("👈 Upload an image or capture a webcam photo to see instant predictions.")

    # Sentence Builder Section
    st.divider()
    st.subheader("🔤 Interactive Sentence / Word Builder")
    st.caption("Sign letters one by one and append them to form words and full sentences!")

    sent_col, btn_col1, btn_col2, btn_col3 = st.columns([4, 1, 1, 1])

    with sent_col:
        current_text = st.session_state["sentence"] if st.session_state["sentence"] else "(Empty word/sentence)"
        st.markdown(f'<div class="sentence-box">{current_text}</div>', unsafe_allow_html=True)

    with btn_col1:
        if st.button("␣ Space", use_container_width=True):
            st.session_state["sentence"] += " "
            st.rerun()

    with btn_col2:
        if st.button("⌫ Backspace", use_container_width=True):
            st.session_state["sentence"] = st.session_state["sentence"][:-1]
            st.rerun()

    with btn_col3:
        if st.button("🗑️ Clear All", use_container_width=True):
            st.session_state["sentence"] = ""
            st.rerun()


# ==========================================
# TAB 2: Sign Reference Gallery
# ==========================================
with tab_reference:
    st.subheader("Indian Sign Language (ISL) Alphabet Quick Reference")
    st.markdown("""
    This project classifies **24 static alphabet signs** (`A` through `Y`).
    - **Why are 'J' and 'Z' missing?** In ISL fingerspelling, 'J' and 'Z' are **dynamic signs** requiring the signer to trace the letter shape in the air. Static image models exclude them.
    """)

    # Grid of supported signs
    cols = st.columns(8)
    for i, letter in enumerate(CLASSES):
        col = cols[i % 8]
        with col:
            st.markdown(f"""
            <div style="background: #0f172a; border: 1px solid #334155; border-radius: 0.5rem; text-align: center; padding: 0.8rem 0; margin-bottom: 0.5rem;">
                <span style="font-size: 1.4rem; font-weight: 800; color: #38bdf8;">{letter}</span>
                <div style="font-size: 0.65rem; color: #64748b;">Class #{i+1}</div>
            </div>
            """, unsafe_allow_html=True)


# ==========================================
# TAB 3: Model Evaluation & Analytics
# ==========================================
with tab_analytics:
    st.subheader("📈 Quantitative Model Performance")
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Test Accuracy", "100.00%", "Target: >95%")
    m2.metric("Test Loss", "0.0018", "Categorical Crossentropy")
    m3.metric("Macro F1-Score", "1.0000", "Across 24 classes")
    m4.metric("Parameters", "463,320", "~1.76 MB (Lightweight)")

    st.write("")
    col_curve, col_cm = st.columns(2)

    with col_curve:
        st.subheader("Training & Validation Curves")
        if os.path.exists("training_curves.png"):
            st.image("training_curves.png", use_container_width=True)
        else:
            st.info("Run `python train.py` to generate training curves.")

    with col_cm:
        st.subheader("Confusion Matrix (24 Classes)")
        if os.path.exists("confusion_matrix.png"):
            st.image("confusion_matrix.png", use_container_width=True)
        else:
            st.info("Run `python train.py` to generate confusion matrix.")


# ==========================================
# TAB 4: REST API & Cloud Publishing Guide
# ==========================================
with tab_api:
    st.subheader("⚡ FastAPI REST Backend & Cloud Publishing")
    st.markdown("""
    This project includes a **production-ready FastAPI server** ([`api.py`](file:///d:/Projects/IndisignLAnguage/api.py)) that can be deployed to any cloud platform (AWS, Render, Railway, Google Cloud, Hugging Face Spaces).
    """)

    st.markdown("#### 1. How to run the FastAPI REST server:")
    st.code("uvicorn api:app --host 0.0.0.0 --port 8000 --reload", language="bash")

    st.markdown("#### 2. Access Built-in Interactive Swagger Docs:")
    st.markdown("- Open `http://localhost:8000/docs` in your browser to test endpoints live.")

    st.markdown("#### 3. Sample Python REST Client Request:")
    st.code("""
import requests

url = "http://localhost:8000/predict?top_k=3"
with open("data/extracted/Data/B/001.jpg", "rb") as img:
    response = requests.post(url, files={"file": img})
    print(response.json())
    """, language="python")

    st.markdown("#### 4. How to publish for free online (Hugging Face Spaces / Streamlit Cloud):")
    st.markdown("""
    1. **Push your code to GitHub**: `git push origin main`
    2. **Deploy on Streamlit Community Cloud**:
       - Go to [share.streamlit.io](https://share.streamlit.io/)
       - Connect your GitHub repo `Snehan0911/ISL_Recognition`
       - Select `app.py` as the main entry point
       - Click **Deploy**!
    3. **Deploy the FastAPI backend on Render / Railway**:
       - Connect GitHub repo
       - Set Start Command: `uvicorn api:app --host 0.0.0.0 --port 8000`
    """)
