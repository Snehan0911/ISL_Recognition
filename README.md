# 🤟 Indian Sign Language (ISL) Recognition Using CNN

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-D00000?logo=keras&logoColor=white)](https://keras.io/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Accuracy](https://img.shields.io/badge/Test%20Accuracy-88.96%25-brightgreen)](#model-performance)

An end-to-end Computer Vision and Deep Learning system designed to recognize **Indian Sign Language (ISL)** alphabet gestures from images using a custom **Convolutional Neural Network (CNN)**.

This project aims to bridge the communication gap between the deaf and hard-of-hearing community and the hearing world by providing accurate, automated sign language recognition.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Project Highlights](#-project-highlights)
- [Dataset Details](#-dataset-details)
- [Data Preprocessing & Augmentation](#-data-preprocessing--augmentation)
- [CNN Model Architecture](#-cnn-model-architecture)
- [Training & Optimization](#-training--optimization)
- [Model Performance & Evaluation](#-model-performance--evaluation)
- [Project Structure](#-project-structure)
- [Installation & Setup](#-installation--setup)
- [How to Run](#-how-to-run)
- [Inference & Predictions](#-inference--predictions)
- [Strengths & Future Roadmap](#-strengths--future-roadmap)
- [License & Acknowledgments](#-license--acknowledgments)

---

## 🌟 Overview

Indian Sign Language (ISL) is predominantly visual-gestural. This project implements a complete, production-grade deep learning workflow that classifies hand sign images into **24 distinct ISL alphabet classes**.

### Supported Classes (24 Alphabet Signs)
```
A, B, C, D, E, F, G, H, I, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y
```
*(Note: 'J' and 'Z' require dynamic motion gestures and are omitted in static image sign datasets).*

---

## ✨ Project Highlights

- **Data Integrity & Duplicate Removal**: Used MD5 cryptographic hashing across all 4,972 images to detect and eliminate duplicate samples, guaranteeing no data contamination.
- **Stratified Data Splitting**: Preserved class proportions across Train (80%), Validation (10%), and Test (10%) splits.
- **Efficient tf.data Pipeline**: Utilized TensorFlow's streaming `tf.data.Dataset` with autotuned prefetching and parallel mapping for low memory footprint and high throughput.
- **Domain-Specific Augmentation**: Applied rotation, zoom, and translation while **intentionally avoiding horizontal flipping**, ensuring hand sign semantic validity is preserved.
- **Class-Imbalance Handling**: Computed balanced class weights from training labels to prevent model bias towards majority classes.
- **Smart Callbacks**: Combined `EarlyStopping`, `ReduceLROnPlateau`, and `ModelCheckpoint` to save the best checkpoint (`best_isl_cnn.keras`).
- **High Generalization**: Achieved **88.96% test accuracy** and **89.74% validation accuracy** with compact model complexity (~424K parameters).

---

## 📊 Dataset Details

| Attribute | Details |
| :--- | :--- |
| **Total Images (Raw)** | 4,972 |
| **Duplicates Detected & Removed** | 1 (`031.jpg` in class A) |
| **Cleaned Dataset** | 4,971 images |
| **Number of Classes** | 24 |
| **Image Resolution** | Standardized to $224 \times 224$ pixels |
| **Color Channels** | 3 (RGB) |
| **Train Set (80%)** | 3,976 images (125 batches of size 32) |
| **Validation Set (10%)** | 497 images (16 batches of size 32) |
| **Test Set (10%)** | 498 images (16 batches of size 32) |

---

## 🔄 Data Preprocessing & Augmentation

1. **Format & Integrity Validation**: Confirmed all images are valid RGB JPEGs with no corrupt files.
2. **Image Resizing & Normalization**: Resized input frames to $224 \times 224 \times 3$ and normalized pixel intensities from $[0, 255]$ to $[0.0, 1.0]$.
3. **Data Augmentation Pipeline**:
   - `RandomRotation(factor=0.08)` (subtle angle variations)
   - `RandomZoom(height_factor=(-0.1, 0.1), width_factor=(-0.1, 0.1))`
   - `RandomTranslation(height_factor=0.08, width_factor=0.08)`
   - *No horizontal flips to avoid flipping left/right hand geometry.*

---

## 🧠 CNN Model Architecture

The custom 4-block Convolutional Neural Network is built for high feature extraction efficiency while preventing overfitting through regularization:

```
Input (224 x 224 x 3)
  │
  ├── Data Augmentation (Rotation, Zoom, Translation)
  │
  ├── [Conv2D (32 filters, 3x3, ReLU, same)] ──> [MaxPooling2D (2x2)]
  │
  ├── [Conv2D (64 filters, 3x3, ReLU, same)] ──> [MaxPooling2D (2x2)]
  │
  ├── [Conv2D (128 filters, 3x3, ReLU, same)] ──> [MaxPooling2D (2x2)]
  │
  ├── [Conv2D (256 filters, 3x3, ReLU, same)] ──> [MaxPooling2D (2x2)]
  │
  ├── [GlobalAveragePooling2D]
  │
  ├── [Dense (128 units, ReLU)]
  │
  ├── [Dropout (0.5)]
  │
  └── [Dense (24 units, Softmax Output)]
```

- **Total Parameters**: 424,408 (1.62 MB)
- **Trainable Parameters**: 424,408
- **Loss Function**: `sparse_categorical_crossentropy`
- **Optimizer**: Adam ($\text{learning rate} = 0.001$)

---

## 🚀 Training & Optimization

The network was trained for 20 epochs using the following callback configuration:
- **`ModelCheckpoint`**: Saved the best weights to `best_isl_cnn.keras` monitoring `val_accuracy`.
- **`EarlyStopping`**: Monitored `val_loss` with patience of 5 epochs and weight restoration.
- **`ReduceLROnPlateau`**: Reduced learning rate by $50\%$ when validation loss plateaued for 2 consecutive epochs.
- **Class Weights**: Dynamically computed via `sklearn.utils.class_weight.compute_class_weight("balanced", ...)` to prevent minority class suppression.

---

## 📈 Model Performance & Evaluation

### Key Results on Unseen Test Set (498 Images)

| Metric | Score |
| :--- | :--- |
| **Test Accuracy** | **88.96%** |
| **Best Validation Accuracy** | **89.74%** |
| **Test Loss** | **0.2750** |
| **Macro Avg F1-Score** | **0.8620** |
| **Weighted Avg F1-Score** | **0.8813** |
| **Multi-Sample Visual Test** | **91.67% (11/12 correct)** |

### Classification Highlights
- Perfect precision and recall ($1.00$) achieved on signs such as **A**, **C**, and **K**.
- High confidence predictions (>95%) across individual test samples.
- Clean confusion matrix showing strong diagonal concentration with minimal inter-class confusion.

---

## 📁 Project Structure

```text
ISL_Recognition/
├── best_isl_cnn.keras            # Trained Keras model checkpoint (best val accuracy)
├── ISL_Recognition.ipynb         # Complete Jupyter notebook (EDA, training, evaluation)
├── predict.py                    # Standalone CLI inference script for new images
├── requirements.txt              # Project dependencies
├── data/
│   └── extracted/
│       └── Data/                 # 24 class folders (A-Y) with sign images
└── README.md                     # Comprehensive documentation
```

---

## 💻 Installation & Setup

### 1. Prerequisites
- Python 3.10 or 3.11 recommended
- Git

### 2. Clone the Repository
```bash
git clone https://github.com/Snehan0911/ISL_Recognition.git
cd ISL_Recognition
```

### 3. Create and Activate a Virtual Environment
```bash
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🏃 How to Run

### Option 1: Explore & Run the Jupyter Notebook
Open the comprehensive notebook to view the step-by-step EDA, training curves, confusion matrix, and observations:
```bash
jupyter notebook ISL_Recognition.ipynb
```

### Option 2: Run CLI Inference on an Image
You can classify any hand sign image directly using `predict.py`:
```bash
python predict.py --image "data/extracted/Data/A/001.jpg"
```
Or specify a custom model checkpoint:
```bash
python predict.py --image "path/to/hand_sign.jpg" --model "best_isl_cnn.keras"
```

---

## 🔮 Strengths & Future Roadmap

### ✅ Current Strengths
1. **Methodical EDA**: Exhaustive data verification, dimension consistency, duplicate detection.
2. **Clean ML Practices**: Strict separation of train/validation/test sets before any fitting or scaling.
3. **Efficient Architecture**: Global Average Pooling prevents dense parameter explosion, keeping the model fast and lightweight (<5 MB).
4. **Rich Documentation**: Complete rationale documented for every modeling choice.

### 🚀 Future Roadmap
- [ ] **Real-Time Webcam Translation**: Integrate OpenCV and MediaPipe Hands to capture hand bounding boxes and run real-time video stream recognition.
- [ ] **Dynamic Gestures**: Add recurrent layers (LSTM / GRU) or Transformer networks to recognize dynamic signs like **J** and **Z**.
- [ ] **Interactive Web UI**: Build a Streamlit or Gradio demo for drag-and-drop web testing.
- [ ] **Edge Deployment**: Convert the model to **TensorFlow Lite (`.tflite`)** or **ONNX** for mobile and Raspberry Pi deployment.
- [ ] **Sign-to-Speech**: Synthesize predicted signs into spoken audio (Text-to-Speech) for two-way communication.

---

## 📄 License & Acknowledgments

- **Dataset**: Indian Sign Language (ISL) Gesture Dataset.
- **Frameworks**: Developed with [TensorFlow](https://www.tensorflow.org/), [Keras](https://keras.io/), [OpenCV](https://opencv.org/), [Scikit-learn](https://scikit-learn.org/), and [Matplotlib](https://matplotlib.org/).
