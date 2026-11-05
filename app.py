"""
app.py: Streamlit Web Application for Indian Sign Language (ISL) Recognition
"""

import os
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

st.set_page_config(
    page_title="ISL Recognition AI",
    page_icon="🤟",
    layout="wide"
)

CLASSES = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y'
]
IMG_SIZE = (128, 128)
MODEL_PATH = "best_isl_cnn.keras"

@st.cache_resource
def load_cached_model():
    """Loads and caches model in memory to keep predictions instant."""
    if not os.path.exists(MODEL_PATH):
        return None
    return tf.keras.models.load_model(MODEL_PATH)

# --- Header Section ---
st.title("🤟 Indian Sign Language (ISL) Recognition AI")
st.markdown(
    "An assistive Deep Learning application powered by a custom **Convolutional Neural Network (CNN)** "
    "to translate Indian Sign Language alphabet gestures into text in real time."
)

# --- Sidebar Information ---
with st.sidebar:
    st.header("📌 Project Details")
    st.markdown("""
    - **Architecture**: 4-Block CNN with Batch Normalization & GAP
    - **Classes**: 24 Sign Letters (`A` to `Y`, excl. dynamic signs `J` & `Z`)
    - **Input Shape**: $128 \\times 128 \\times 3$
    - **Framework**: TensorFlow / Keras
    - **Test Accuracy**: **100%** on 482 test images
    """)
    st.divider()
    st.info("💡 **Tip**: Ensure good lighting and that your hand gesture fills most of the frame for best predictions.")

model = load_cached_model()

if model is None:
    st.error(f"❌ Model file `{MODEL_PATH}` not found! Please run `python train.py` first.")
    st.stop()

# --- Main App Layout ---
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📤 Provide Sign Gesture")
    upload_method = st.radio("Choose Input Method:", ["Upload Image File", "Use Webcam Camera"], horizontal=True)

    input_image = None
    if upload_method == "Upload Image File":
        uploaded_file = st.file_uploader("Upload a hand sign photo (JPG, PNG, JPEG):", type=["jpg", "png", "jpeg"])
        if uploaded_file is not None:
            input_image = Image.open(uploaded_file).convert("RGB")
    else:
        camera_file = st.camera_input("Take a photo of your hand sign:")
        if camera_file is not None:
            input_image = Image.open(camera_file).convert("RGB")

    if input_image:
        st.image(input_image, caption="Input Hand Gesture", use_container_width=True)

with col2:
    st.subheader("🎯 Prediction Result")
    if input_image is not None:
        # Preprocessing
        img_resized = input_image.resize(IMG_SIZE)
        img_array = np.array(img_resized, dtype=np.float32)
        img_batch = np.expand_dims(img_array, axis=0)

        # Inference
        with st.spinner("Analyzing hand geometry..."):
            preds = model.predict(img_batch, verbose=0)[0]

        top_indices = np.argsort(preds)[::-1][:3]
        top_class = CLASSES[top_indices[0]]
        top_conf = preds[top_indices[0]] * 100

        # Display Top Prediction Card
        st.success(f"### Recognized Sign: **{top_class}**")
        st.metric(label="Model Confidence", value=f"{top_conf:.2f}%")

        # Top 3 Confidence Bar Chart
        st.write("#### Top 3 Predicted Classes:")
        top_labels = [CLASSES[idx] for idx in top_indices]
        top_probs = [float(preds[idx] * 100) for idx in top_indices]
        chart_data = dict(zip(top_labels, top_probs))
        st.bar_chart(chart_data)

    else:
        st.info("👈 Upload an image or capture via webcam to see predictions.")

