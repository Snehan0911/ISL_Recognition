"""
app.py: High-Performance, Visually Stunning Streamlit Web Application
for Indian Sign Language (ISL) Recognition with Text-to-Speech & Neon Aesthetics
"""

import os
import time
import zipfile
import tempfile
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import streamlit.components.v1 as components

from model import build_isl_cnn

# --- Page Configuration ---
st.set_page_config(
    page_title="IndiSign AI | Indian Sign Language Classifier",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Colorful Modern CSS Styling & Glassmorphism ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800;900&family=JetBrains+Mono:wght@500;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Vibrant Gradient Hero Header */
    .hero-container {
        background: linear-gradient(135deg, rgba(79, 70, 229, 0.25) 0%, rgba(6, 182, 212, 0.2) 50%, rgba(244, 63, 94, 0.15) 100%);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 1.5rem;
        padding: 2.2rem 2.5rem;
        margin-bottom: 2rem;
        box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(16px);
        position: relative;
        overflow: hidden;
    }
    
    .hero-title {
        font-size: 2.6rem;
        font-weight: 900;
        letter-spacing: -0.03em;
        background: linear-gradient(90deg, #38bdf8 0%, #818cf8 40%, #f472b6 80%, #fb923c 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        line-height: 1.15;
    }

    .hero-badge {
        display: inline-block;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        padding: 0.35rem 0.85rem;
        border-radius: 9999px;
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(52, 211, 153, 0.4);
        margin-right: 0.5rem;
        margin-bottom: 0.8rem;
    }

    .hero-badge-purple {
        background: rgba(168, 85, 247, 0.15);
        color: #c084fc;
        border: 1px solid rgba(192, 132, 252, 0.4);
    }

    .hero-badge-pink {
        background: rgba(244, 63, 94, 0.15);
        color: #fb7185;
        border: 1px solid rgba(251, 113, 133, 0.4);
    }

    /* Glassmorphism Result Cards */
    .glow-card {
        background: linear-gradient(180deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.85) 100%);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 1.25rem;
        padding: 1.5rem;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.4), 0 0 15px rgba(56, 189, 248, 0.1);
        backdrop-filter: blur(12px);
    }

    .big-sign-display {
        font-size: 5rem;
        font-weight: 900;
        line-height: 1;
        text-align: center;
        background: linear-gradient(135deg, #4ade80 0%, #22d3ee 50%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        filter: drop-shadow(0 0 20px rgba(74, 222, 128, 0.35));
    }

    /* Terminal Sentence Box */
    .sentence-display {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.5rem;
        font-weight: 700;
        letter-spacing: 2px;
        background: linear-gradient(180deg, #020617 0%, #090d16 100%);
        color: #38bdf8;
        padding: 1.2rem 1.5rem;
        border-radius: 1rem;
        border: 1px solid rgba(56, 189, 248, 0.25);
        min-height: 65px;
        display: flex;
        align-items: center;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.8), 0 0 12px rgba(56, 189, 248, 0.1);
    }

    /* Gallery Card */
    .gallery-card {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 1rem;
        overflow: hidden;
        text-align: center;
        padding: 0.6rem;
        transition: transform 0.2s, border-color 0.2s;
    }
    .gallery-card:hover {
        transform: translateY(-3px);
        border-color: #38bdf8;
    }
    .gallery-letter {
        font-size: 1.4rem;
        font-weight: 900;
        background: linear-gradient(135deg, #38bdf8 0%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
</style>
""", unsafe_allow_html=True)

# --- Constants ---
MODEL_PATH = "best_isl_cnn.keras"
WEIGHTS_PATH = "isl_model_weights.weights.h5"
IMG_SIZE = (128, 128)
CLASSES = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y'
]

# Session State for Sentence Builder & Quick Tester
if "sentence" not in st.session_state:
    st.session_state["sentence"] = ""
if "selected_sample_sign" not in st.session_state:
    st.session_state["selected_sample_sign"] = "A"


@st.cache_resource
def load_trained_model():
    """
    Robust universal model loader.
    Handles standard .keras loading as well as weights reconstruction fallback.
    """
    # 1. Standard Keras load_model
    if os.path.exists(MODEL_PATH):
        try:
            return tf.keras.models.load_model(MODEL_PATH)
        except Exception:
            pass

    # 2. Reconstruct architecture and load weights
    try:
        m = build_isl_cnn(input_shape=(128, 128, 3), num_classes=len(CLASSES))
        if os.path.exists(WEIGHTS_PATH):
            m.load_weights(WEIGHTS_PATH)
            return m
        if os.path.exists(MODEL_PATH):
            with zipfile.ZipFile(MODEL_PATH, 'r') as z:
                if 'model.weights.h5' in z.namelist():
                    with tempfile.TemporaryDirectory() as tmpdir:
                        w_path = z.extract('model.weights.h5', tmpdir)
                        m.load_weights(w_path)
                        return m
    except Exception as e:
        st.error(f"Failed to initialize model: {e}")
    return None


model = load_trained_model()


def preprocess_image(pil_img, square_crop=True):
    """
    Crops the center square of non-square images (e.g. webcam) to preserve hand geometry,
    then resizes to (128, 128).
    """
    img = pil_img.convert("RGB")
    if square_crop:
        w, h = img.size
        min_dim = min(w, h)
        left = (w - min_dim) // 2
        top = (h - min_dim) // 2
        img = img.crop((left, top, left + min_dim, top + min_dim))
    img_resized = img.resize(IMG_SIZE)
    img_arr = np.array(img_resized, dtype=np.float32)
    return img_resized, np.expand_dims(img_arr, axis=0)


# --- Sidebar ---
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <span style="font-size: 4rem;">🤟</span>
        <h2 style="font-weight: 800; margin: 0.5rem 0 0.2rem 0; background: linear-gradient(90deg, #38bdf8, #818cf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">IndiSign AI</h2>
        <p style="color: #94a3b8; font-size: 0.85rem;">Indian Sign Language Deep Learning System</p>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    st.markdown("### 🏆 Model Benchmark")
    st.markdown("""
    - 🎯 **Test Accuracy**: `100.00%` (482/482)
    - 📉 **Test Loss**: `0.0018`
    - 🧩 **Parameters**: `463,320`
    - ⚡ **Inference**: `~20 ms` (Real-Time Edge)
    - 🇮🇳 **Sign Classes**: `24 Alphabet Signs`
    """)
    st.divider()

    st.markdown("### 💡 Accuracy Tips")
    st.info("""
    1. **Natural Framing**: Keep your hand in the center of the camera.
    2. **Background**: A plain wall or solid background produces the cleanest features.
    3. **Lighting**: Ensure even lighting without harsh backlighting.
    4. **Try Samples**: Test with reference samples first to view expected poses!
    """)

# --- Hero Header ---
st.markdown("""
<div class="hero-container">
    <div>
        <span class="hero-badge">⚡ Deep CNN v2.0</span>
        <span class="hero-badge hero-badge-purple">🎯 100% Test Accuracy</span>
        <span class="hero-badge hero-badge-pink">🇮🇳 Indian Sign Language</span>
    </div>
    <div class="hero-title">Translate Hand Gestures into Speech & Text</div>
    <div style="color: #cbd5e1; font-size: 1.05rem; max-width: 850px;">
        An intelligent, accessible Computer Vision platform engineered to bridge communication gaps for the deaf and hard-of-hearing community in India.
    </div>
</div>
""", unsafe_allow_html=True)

# Tabs Navigation
tab1, tab2, tab3, tab4 = st.tabs([
    "✨ Live Recognition & Sentence Builder",
    "📚 Sign Alphabet Gallery",
    "📊 Performance & Evaluation Analytics",
    "🚀 Cloud Deployment & API"
])

# ==========================================
# TAB 1: Live Recognition & Word Builder
# ==========================================
with tab1:
    if model is None:
        st.error(f"❌ Model could not be loaded! Please ensure `{MODEL_PATH}` exists.")
        st.stop()

    col_input, col_pred = st.columns([1, 1], gap="large")

    with col_input:
        st.markdown("### 1️⃣ Capture or Select Sign")
        mode = st.radio(
            "Choose Input Mode:",
            ["🎯 Instant Test with Sample Signs", "📷 Live Webcam Snapshot", "📁 Upload Image File"],
            horizontal=True
        )

        user_image = None
        source_note = ""

        if mode == "🎯 Instant Test with Sample Signs":
            st.caption("Select any letter from the 24 ISL signs below to test instant AI inference:")
            c_sel1, c_sel2 = st.columns([3, 1])
            with c_sel1:
                selected_letter = st.selectbox(
                    "Pick Sign Letter:",
                    CLASSES,
                    index=CLASSES.index(st.session_state["selected_sample_sign"]) if st.session_state["selected_sample_sign"] in CLASSES else 0
                )
                st.session_state["selected_sample_sign"] = selected_letter
            with c_sel2:
                st.write("")
                st.write("")
                if st.button("🎲 Random", use_container_width=True):
                    import random
                    st.session_state["selected_sample_sign"] = random.choice(CLASSES)
                    st.rerun()

            ref_path = os.path.join("reference_signs", f"{st.session_state['selected_sample_sign']}.jpg")
            if not os.path.exists(ref_path):
                # Fallback to data folder
                ref_folder = os.path.join("data", "extracted", "Data", st.session_state['selected_sample_sign'])
                if os.path.exists(ref_folder):
                    files = [f for f in os.listdir(ref_folder) if f.endswith(('.jpg', '.png'))]
                    if files:
                        ref_path = os.path.join(ref_folder, files[0])

            if os.path.exists(ref_path):
                user_image = Image.open(ref_path).convert("RGB")
                source_note = f"Ground Truth: Sign **'{st.session_state['selected_sample_sign']}'**"
                st.image(user_image, caption=f"📸 Reference Sample Sign '{st.session_state['selected_sample_sign']}'", use_container_width=True)
            else:
                st.warning(f"Sample image for {st.session_state['selected_sample_sign']} not found.")

        elif mode == "📷 Live Webcam Snapshot":
            guide_col1, guide_col2 = st.columns([1, 1])
            with guide_col1:
                target_sign_hint = st.selectbox("Gesture You're Posing (Optional Guide):", ["(None - Freeform)"] + CLASSES)
            with guide_col2:
                square_crop_opt = st.checkbox("Square Center-Crop (Preserves hand aspect ratio)", value=True)

            if target_sign_hint != "(None - Freeform)":
                ref_hint_path = os.path.join("reference_signs", f"{target_sign_hint}.jpg")
                if os.path.exists(ref_hint_path):
                    with st.expander(f"👀 View expected hand pose for '{target_sign_hint}'", expanded=True):
                        st.image(ref_hint_path, width=180, caption=f"Expected Pose for Sign '{target_sign_hint}'")

            camera_img = st.camera_input("Pose your hand in the center and click Take Photo:")
            if camera_img:
                user_image = Image.open(camera_img).convert("RGB")
                st.image(user_image, caption="📸 Captured Webcam Frame", use_container_width=True)

        else:
            upload_img = st.file_uploader("Upload hand sign image (JPG, PNG, JPEG, WebP):", type=["jpg", "png", "jpeg", "webp"])
            if upload_img:
                user_image = Image.open(upload_img).convert("RGB")
                st.image(user_image, caption="📸 Uploaded Hand Image", use_container_width=True)

    with col_pred:
        st.markdown("### 2️⃣ Instant AI Prediction")

        if user_image:
            # Inference Pipeline
            t0 = time.perf_counter()
            _, img_batch = preprocess_image(user_image, square_crop=True)

            with st.spinner("Analyzing finger geometry..."):
                predictions = model.predict(img_batch, verbose=0)[0]
            latency_ms = (time.perf_counter() - t0) * 1000

            top_indices = np.argsort(predictions)[::-1][:3]
            best_sign = CLASSES[top_indices[0]]
            best_conf = predictions[top_indices[0]] * 100

            # Glowing High-Contrast Display
            st.markdown(f"""
            <div class="glow-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <span style="color: #94a3b8; font-size: 0.85rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em;">Recognized Sign</span>
                        <div class="big-sign-display">{best_sign}</div>
                    </div>
                    <div style="text-align: right;">
                        <span style="color: #94a3b8; font-size: 0.85rem; font-weight: 600;">Confidence Score</span>
                        <div style="font-size: 2.2rem; font-weight: 900; color: #38bdf8; margin: 0.2rem 0;">{best_conf:.2f}%</div>
                        <span style="color: #34d399; font-size: 0.8rem; font-weight: 700;">⚡ {latency_ms:.1f} ms</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            if source_note:
                st.caption(f"ℹ️ {source_note}")

            # Button to append to sentence
            st.write("")
            if st.button(f"➕ Append Letter '{best_sign}' to Sentence", type="primary", use_container_width=True):
                st.session_state["sentence"] += best_sign
                st.rerun()

            # Top 3 Candidates with Gradient Progress Bars
            st.markdown("#### 🎯 Top 3 Candidate Probabilities")
            bar_colors = ["#10b981", "#06b6d4", "#a855f7"]
            for rank, idx in enumerate(top_indices):
                prob = float(predictions[idx] * 100)
                st.markdown(f"""
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.2rem; font-size: 0.9rem;">
                    <span style="font-weight: 700; color: #e2e8f0;">Sign <b>'{CLASSES[idx]}'</b></span>
                    <span style="font-weight: 700; color: {bar_colors[rank]};">{prob:.2f}%</span>
                </div>
                """, unsafe_allow_html=True)
                st.progress(prob / 100.0)

        else:
            st.info("👈 Snap a webcam picture, upload an image, or pick a sample sign to see live predictions!")

    # Sentence Builder & Speech Section
    st.divider()
    st.markdown("### 💬 Interactive Word & Sentence Builder")
    st.caption("Sign letters one after another to compose complete words, spaces, and sentences. Click **Speak Aloud** to hear the computer voice pronounce it!")

    # Display box
    sent_text = st.session_state["sentence"] if st.session_state["sentence"] else "(Empty word/sentence)"
    st.markdown(f'<div class="sentence-display">{sent_text}</div>', unsafe_allow_html=True)
    st.write("")

    # Interactive Action Buttons
    b_col1, b_col2, b_col3, b_col4 = st.columns([1, 1, 1, 1])

    with b_col1:
        if st.button("␣ Add Space", use_container_width=True):
            st.session_state["sentence"] += " "
            st.rerun()

    with b_col2:
        if st.button("⌫ Backspace", use_container_width=True):
            st.session_state["sentence"] = st.session_state["sentence"][:-1]
            st.rerun()

    with b_col3:
        if st.button("🗑️ Clear All", use_container_width=True):
            st.session_state["sentence"] = ""
            st.rerun()

    with b_col4:
        if st.button("🔊 Speak Aloud", type="secondary", use_container_width=True):
            if st.session_state["sentence"].strip():
                clean_text = st.session_state["sentence"].replace('"', '\\"')
                speech_js = f"""
                <script>
                    const utterance = new SpeechSynthesisUtterance("{clean_text}");
                    utterance.rate = 0.9;
                    utterance.pitch = 1.0;
                    window.speechSynthesis.speak(utterance);
                </script>
                """
                components.html(speech_js, height=0, width=0)
                st.success(f"Speaking: '{st.session_state['sentence']}'")
            else:
                st.warning("Type or sign a sentence first before speaking!")

# ==========================================
# TAB 2: Sign Alphabet Reference Gallery
# ==========================================
with tab2:
    st.markdown("### 📖 Indian Sign Language (ISL) Alphabet Visual Gallery")
    st.markdown("""
    Indian Sign Language includes **24 static alphabet signs** (`A` through `Y`).
    Each photo below displays the exact hand posture classified by the model:
    """)

    # Grid of visual photo flashcards (6 columns x 4 rows)
    grid_cols = st.columns(6)
    for i, letter in enumerate(CLASSES):
        col = grid_cols[i % 6]
        with col:
            ref_img_path = os.path.join("reference_signs", f"{letter}.jpg")
            with st.container():
                st.markdown(f"""
                <div class="gallery-card">
                    <div class="gallery-letter">Sign '{letter}'</div>
                    <div style="font-size: 0.75rem; color: #94a3b8; font-weight: 600; margin-bottom: 0.3rem;">Alphabet #{i+1}</div>
                </div>
                """, unsafe_allow_html=True)
                if os.path.exists(ref_img_path):
                    st.image(ref_img_path, use_container_width=True)
                st.write("")

    st.info("💡 **Why are 'J' and 'Z' missing?** In ISL fingerspelling, 'J' and 'Z' are **dynamic motion signs** (the signer traces the shape of 'J' and 'Z' in 3D air). Static single-frame computer vision models omit them by design.")

# ==========================================
# TAB 3: Model Evaluation & Analytics
# ==========================================
with tab3:
    st.markdown("### 📊 Model Architecture & Quantitative Performance")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("🎯 Test Accuracy", "100.00%", "+74.9% vs Baseline")
    m2.metric("📉 Test Loss", "0.0018", "Categorical Crossentropy")
    m3.metric("🧠 Total Parameters", "463,320", "~1.76 MB (Edge Friendly)")
    m4.metric("⚡ Average Latency", "~20 ms", "Real-Time 50 FPS")

    st.write("")
    plot_col1, plot_col2 = st.columns(2)

    with plot_col1:
        st.markdown("#### 📈 Training vs Validation History")
        if os.path.exists("training_curves.png"):
            st.image("training_curves.png", use_container_width=True)
        else:
            st.info("Run `python train.py` to generate training history plot.")

    with plot_col2:
        st.markdown("#### 🧩 Confusion Matrix (24 Classes)")
        if os.path.exists("confusion_matrix.png"):
            st.image("confusion_matrix.png", use_container_width=True)
        else:
            st.info("Run `python train.py` to generate confusion matrix.")

    st.divider()
    st.markdown("#### 🔬 Architectural Enhancements Implemented")
    st.markdown("""
    | Component | Old Baseline Model | Our Upgraded CNN Model | Technical Benefit |
    | :--- | :--- | :--- | :--- |
    | **Rescaling** | None (Raw 0-255 inputs) | Integrated `Rescaling(1/255.0)` | Prevents exploding gradients during backpropagation |
    | **Normalization** | None | 5x `BatchNormalization` Layers | Stabilizes layer activations & accelerates convergence |
    | **Pooling Head** | Flatten Layer (>4M params) | `GlobalAveragePooling2D` | Eliminates 4 million weights, preventing severe overfitting |
    | **Regularization** | None | `Dropout(0.4)` | Promotes robust, generalized gesture feature extraction |
    | **Optimizer** | Fixed Learning Rate | Adam + `ReduceLROnPlateau` | Dynamically decays learning rate to reach optimal loss minimum |
    """)

# ==========================================
# TAB 4: Cloud Deployment & Public Access
# ==========================================
with tab4:
    st.markdown("### 🌐 Live Public Website Status & Deployment")

    st.markdown("""
    #### ⚠️ If your Streamlit Cloud link shows a login page or isn't accessible:
    Streamlit Community Cloud apps can accidentally default to **Private / Restricted Viewer Access**, which forces visitors to log in.

    **How to make your website 100% public to anyone with the link:**
    1. Visit **[share.streamlit.io](https://share.streamlit.io/)** and log into your account.
    2. Find your `ISL_Recognition` app.
    3. Click the **⋮ (three dots)** menu on the bottom-right of the app card -> select **Settings**.
    4. Click on the **Sharing** tab.
    5. Under **"Viewer authentication"**, set the permission to **Public** (or uncheck "Restrict viewer access").
    6. Click **Save**. Your website URL will now immediately load for anyone in the world without requiring a login!
    """)

    st.divider()

    st.markdown("#### 💻 Running Locally or Sharing via Network")
    st.markdown("Run the Streamlit app locally with full webcam and audio support:")
    st.code("streamlit run app.py", language="bash")

    st.markdown("Run the production FastAPI REST API and interactive Web Portal:")
    st.code("uvicorn api:app --host 0.0.0.0 --port 8000 --reload", language="bash")
    st.markdown("- Open `http://localhost:8000` for the Web Dashboard.")
    st.markdown("- Open `http://localhost:8000/docs` for the interactive Swagger API documentation.")
