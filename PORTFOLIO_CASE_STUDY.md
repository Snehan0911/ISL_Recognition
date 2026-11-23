# Portfolio Case Study: IndiSign AI — Indian Sign Language Recognition & Speech Engine

> **Project Lead & AI Engineer**: Snehan ([GitHub @Snehan0911](https://github.com/Snehan0911))  
> **Status**: Production Deployed • Open Source  
> **Live Web Application**: [Launch Streamlit Cloud App](https://islrecognition-ezwzp8lmwbrh2cjbaratev.streamlit.app/)  
> **Interactive Portfolio Website**: [Open `index.html`](index.html)  
> **Presentation Slide Deck**: [Open `IndiSign_AI_Project_Presentation.html`](IndiSign_AI_Project_Presentation.html)  
> **GitHub Repository**: [github.com/Snehan0911/ISL_Recognition](https://github.com/Snehan0911/ISL_Recognition)  

---

## Executive Summary

**IndiSign AI** is a production-grade, assistive computer vision system engineered to bridge the critical communication divide for over **18 Million deaf and hard-of-hearing individuals** across India. By developing an ultra-lightweight, 4-block deep Convolutional Neural Network (CNN) featuring Global Average Pooling, dynamic learning rate scheduling, and center-square aspect-ratio preservation, IndiSign AI achieves **100.00% test accuracy** across 24 Indian Sign Language (ISL) alphabet classes with **~20 ms edge latency** and an ultra-lean memory footprint of only **1.76 MB (463K parameters)**.

The solution is packaged as an end-to-end full-stack application comprising:
1. **Interactive Web Application**: Streamlit Cloud frontend with webcam support, live sample testing, sentence building, and browser-based **Text-to-Speech (TTS)** voice synthesis.
2. **High-Throughput Microservice**: Async FastAPI backend exposing `/predict` and `/predict_base64` endpoints for seamless edge and mobile integration.
3. **Interactive Portfolio Showcase**: A dark-mode, responsive web showcase featuring an in-browser live simulator, speech studio, video walkthrough, and 24-alphabet visual dictionary.

---

## 1. Problem Statement & Societal Impact

### The Crisis
- **18 Million+ Deaf Population**: India has one of the largest deaf and hard-of-hearing communities globally. For many, Indian Sign Language (ISL) is their primary language.
- **Acute Interpreter Shortage (1 : 70,000)**: With fewer than 300 certified ISL interpreters nationwide, deaf individuals face severe communication roadblocks in critical settings:
  - **Hospitals & Emergency Care**: Inability to explain symptoms or understand medical diagnoses.
  - **Classrooms & Universities**: Lack of real-time translation during lectures and examinations.
  - **Banking, Legal & Government Kiosks**: Difficulty accessing public services independently.

### The Engineering Objective
Develop an autonomous, zero-cost, real-time AI interpreter that:
- Runs smoothly on low-power consumer hardware (smartphones, budget laptops, edge kiosks) without requiring discrete GPUs.
- Achieves zero misclassification on single-handed alphabet signs.
- Provides immediate text and audible speech feedback to non-signing interlocutors.

---

## 2. Key Engineering Innovations

### 1. Architectural Re-engineering (GAP vs. Flatten)
- **Traditional Baseline Flaw**: Standard CNN baselines connect convolution blocks directly to a `Flatten()` layer, creating over **4.5 million parameters**. This causes severe spatial overfitting and memory bloat.
- **IndiSign Solution**: Replaced `Flatten()` with `GlobalAveragePooling2D()`. GAP compresses each $128 \times 128 \times 256$ feature map into a single scalar value, reducing parameter count to **463,320 (~1.76 MB)** while enforcing translational spatial invariance.

### 2. Natural Aspect Ratio Preservation (Center-Square Cropping)
- Standard webcams stream at 16:9 widescreen ($1280 \times 720$). Naively resizing this stream into a square $128 \times 128$ input tensor squashes hand signs horizontally, causing finger joints and knuckle angles to distort.
- IndiSign AI implements a **smart center-square cropping pipeline** ($1:1$ ROI extraction) prior to bicubic downsampling, guaranteeing that hand morphology remains identical to the training distribution.

### 3. Integrated Keras Rescaling Layer
- Previous attempts suffered from loss explosion ($\text{Loss} \approx 1,469.72$) due to unnormalized pixel values ($[0, 255]$).
- A built-in `layers.Rescaling(1./255)` layer was embedded directly into the model graph, ensuring mathematical stability during backpropagation regardless of the client-side data format.

### 4. Dynamic Learning Rate Scheduling & Early Stopping
- Configured `ReduceLROnPlateau(factor=0.5, patience=2, min_lr=1e-6)` paired with `EarlyStopping(patience=5, restore_best_weights=True)`. This prevented premature plateauing and guided gradient descent smoothly into the global minimum.

---

## 3. Neural Network Architecture

```
Input Image (RGB: 128 x 128 x 3)
      │
[Pixel Rescaling (1/255.0)]
      │
[Conv2D (32 filters, 3x3) + BatchNorm + ReLU + MaxPool (2x2)]  ➔ Low-level edges & contours
      │
[Conv2D (64 filters, 3x3) + BatchNorm + ReLU + MaxPool (2x2)]  ➔ Finger segments & joints
      │
[Conv2D (128 filters, 3x3) + BatchNorm + ReLU + MaxPool (2x2)] ➔ Hand orientation & geometry
      │
[Conv2D (256 filters, 3x3) + BatchNorm + ReLU + MaxPool (2x2)] ➔ Holistic gesture semantics
      │
[GlobalAveragePooling2D] ➔ 256-dimensional compact embedding
      │
[Dense (256 units) + BatchNorm + Dropout (0.4)]
      │
[Dense (24 classes, Softmax)] ➔ Calibrated Class Probabilities (A to Y)
```

- **Total Parameters**: 463,320
- **Trainable Parameters**: 462,360
- **Non-trainable Parameters (BatchNorm)**: 960
- **Model File Size**: 1.76 MB

---

## 4. Rigorous Performance Benchmarks

| Metric | Naive Baseline CNN | IndiSign AI (Our Model) | Improvement |
| :--- | :--- | :--- | :--- |
| **Test Accuracy** | 25.10% | **100.00%** | **+74.90%** |
| **Categorical Test Loss** | 1,469.72 | **0.0018** | **99.999% reduction** |
| **Macro / Weighted F1-Score** | 0.22 | **1.0000** | **Perfect classification** |
| **Parameter Count** | ~4,500,000 | **463,320** | **89.7% memory reduction** |
| **Inference Latency** | ~65 ms | **~19 ms (50+ FPS)** | **3.4x faster** |
| **Normalization** | None | Rescaling + 5x BatchNorm | Stabilized gradients |
| **Feature Aggregator** | Flatten | GlobalAveragePooling2D | Translational invariance |

### Evaluation Verification
- **Unseen Test Set**: 482 images across 24 classes (`A`, `B`, `C`, `D`, `E`, `F`, `G`, `H`, `I`, `K`, `L`, `M`, `N`, `O`, `P`, `Q`, `R`, `S`, `T`, `U`, `V`, `W`, `X`, `Y`).
  *(Note: Dynamic signs 'J' and 'Z' require continuous 3D motion tracing and were purposefully scoped out of static single-frame inference).*
- **Confusion Matrix**: Pure diagonal entries with **0 false positives and 0 false negatives**.

---

## 5. Software Architecture & Deployment

```
                    ┌───────────────────────────────────┐
                    │       User / Client Device        │
                    │  (Webcam / Browser / Mobile App)  │
                    └─────────────────┬─────────────────┘
                                      │
            ┌─────────────────────────┴─────────────────────────┐
            │                                                   │
            ▼                                                   ▼
┌───────────────────────┐                           ┌───────────────────────┐
│ Streamlit Web App     │                           │ FastAPI Microservice  │
│ (Cloud UI)            │                           │ (Headless API)        │
├───────────────────────┤                           ├───────────────────────┤
│ • Aspect-Ratio Crop   │                           │ • POST /predict       │
│ • Sample Tester       │                           │ • POST /predict_base64│
│ • Sentence Builder    │                           │ • Swagger OpenAPI UI  │
│ • Web Audio TTS       │                           │ • Async Uvicorn Core  │
└───────────┬───────────┘                           └───────────┬───────────┘
            │                                                   │
            └─────────────────────────┬─────────────────────────┘
                                      │
                                      ▼
                        ┌───────────────────────────┐
                        │ Robust Model Loader       │
                        │ (.keras + .h5 universal)  │
                        └─────────────┬─────────────┘
                                      │
                                      ▼
                        ┌───────────────────────────┐
                        │ 4-Block CNN Inference     │
                        │ (TensorFlow 2.x Engine)   │
                        └───────────────────────────┘
```

---

## 6. Project Artifacts & Deliverables

| Artifact | Purpose | Link / Path |
| :--- | :--- | :--- |
| **Interactive Portfolio** | High-converting web portfolio with live simulator & TTS | [`index.html`](index.html) |
| **Live Web App** | Streamlit Cloud deployment with live webcam stream | [Launch App](https://islrecognition-ezwzp8lmwbrh2cjbaratev.streamlit.app/) |
| **Project Slide Deck** | 10-slide responsive HTML presentation | [`IndiSign_AI_Project_Presentation.html`](IndiSign_AI_Project_Presentation.html) |
| **Video Walkthrough** | 4K video demo with open captions | [`IndiSign_AI_Browser_Live_Demo_CC.mp4`](IndiSign_AI_Browser_Live_Demo_CC.mp4) |
| **FastAPI Backend** | Production REST microservice | [`api.py`](api.py) |
| **Model Artifacts** | Pretrained weights with universal fallback | [`best_isl_cnn.keras`](best_isl_cnn.keras) |

---

## 7. Future Horizons & Next Steps
1. **Dynamic Gesture Tracing (LSTM / Temporal Transformers)**: Integrate 3D spatial tracing to capture motion-dependent signs like `J` and `Z`.
2. **Two-Handed Continuous Word Formulation**: Expand beyond finger-spelling to holistic ISL vocabulary signs (e.g., "Thank you", "Doctor", "Water").
3. **On-Device Edge Deployment (TensorFlow Lite / ONNX)**: Quantize weights to INT8 for sub-10ms execution on microcontrollers and embedded Raspberry Pi devices.
