# Indian Sign Language (ISL) Recognition using Deep CNN

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.21-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-Deployment-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Test Accuracy](https://img.shields.io/badge/Test%20Accuracy-100%25-brightgreen)](#model-performance--evaluation)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

An assistive vision-based AI system engineered to classify Indian Sign Language (ISL) alphabet gestures from static images into corresponding textual characters in real time. Built with a custom 4-block Convolutional Neural Network (CNN) featuring Batch Normalization, Global Average Pooling, and Dropout regularization.

---

## 📌 Problem Statement & Business Context
- **The Challenge**: Indian Sign Language (ISL) is the primary mode of communication for millions of deaf and hard-of-hearing individuals across India. The acute scarcity of certified ISL interpreters creates significant communication barriers in educational, healthcare, and professional environments.
- **The Solution**: An automated, lightweight, edge-deployable deep learning recognition model that translates hand gestures into text with high confidence and low latency.

---

## 📊 Dataset & Exploratory Data Analysis (EDA)
- **Dataset Source**: Indian Sign Language Capstone Dataset (4,972 total images across 24 classes).
- **Classes**: 24 alphabet signs: `A`, `B`, `C`, `D`, `E`, `F`, `G`, `H`, `I`, `K`, `L`, `M`, `N`, `O`, `P`, `Q`, `R`, `S`, `T`, `U`, `V`, `W`, `X`, `Y`.
- **Note on Dynamic Signs ('J' & 'Z')**: Omitted because 'J' and 'Z' require dynamic motion tracing in 3D space, which cannot be represented in single static image frames.
- **Split Ratio**: 80% Training (3,978 images), 10% Validation (497 images), 10% Held-Out Testing (482 images).

![Dataset Class Distribution](isl_dataset_distribution.png)

---

## 🧠 Model Architecture

The architecture is custom-designed to balance feature extraction capacity with parameter efficiency:

```
Input Image (128 x 128 x 3)
      │
[Pixel Rescaling (1/255.0)]
      │
[Conv2D (32 filters, 3x3) + BatchNorm + ReLU + MaxPool (2x2)]
      │
[Conv2D (64 filters, 3x3) + BatchNorm + ReLU + MaxPool (2x2)]
      │
[Conv2D (128 filters, 3x3) + BatchNorm + ReLU + MaxPool (2x2)]
      │
[Conv2D (256 filters, 3x3) + BatchNorm + ReLU + MaxPool (2x2)]
      │
[GlobalAveragePooling2D] ➔ 256 feature vector
      │
[Dense (256 units) + BatchNorm + Dropout (0.4)]
      │
[Dense (24 classes, Softmax)] ➔ Class Probabilities
```

- **Total Parameters**: 463,320 (~1.76 MB)
- **Global Average Pooling**: Replaces traditional Flatten layers, preventing over 4 million redundant parameters and eliminating spatial overfitting.

---

## 📈 Model Performance & Evaluation

The upgraded model achieved near-optimal convergence:
- **Test Set Accuracy**: **100.00%** on 482 unseen test images.
- **Test Loss**: **0.0018** (Categorical Cross-Entropy).
- **Macro / Weighted F1-Score**: **1.0000** across all 24 individual classes.

| Metric | Previous Baseline Model | Our Upgraded CNN Model |
| :--- | :--- | :--- |
| **Test Accuracy** | 25.10% | **100.00%** |
| **Test Loss** | 1,469.72 | **0.0018** |
| **Normalization** | None (exploding loss) | Built-in Rescaling Layer |
| **Regularization** | None | BatchNorm + Dropout (0.4) |
| **Convergence** | Diverged | Dynamic LR (`ReduceLROnPlateau`) |

### Training History Curves
![Training Curves](training_curves.png)

### Confusion Matrix
![Confusion Matrix](confusion_matrix.png)

---

## 🚀 Quick Start Guide

### 1. Environment Setup
```powershell
# Clone the repository
git clone https://github.com/your-username/ISL_Recognition.git
cd ISL_Recognition

# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Dataset EDA
```powershell
python eda_analysis.py
```

### 3. Train the Model
```powershell
python train.py
```

### 4. Run CLI Prediction on Any Image
```powershell
python predict.py --image "data/extracted/Data/B/001.jpg"
```

### 5. Launch the Streamlit Web Application
```powershell
streamlit run app.py
```

---

## 📂 Project Structure

```
IndisignLAnguage/
│
├── data/extracted/Data/         # Raw dataset sorted by class (A to Y)
├── eda_analysis.py              # Exploratory Data Analysis & visualizer
├── data_loader.py               # Preprocessing, data augmentation & caching pipeline
├── model.py                     # Custom 4-block CNN architecture definition
├── train.py                     # Training loop with callbacks & evaluation
├── predict.py                   # Standalone CLI inference script
├── app.py                       # Interactive Streamlit web interface
├── best_isl_cnn.keras           # Trained optimal model weights
├── isl_dataset_distribution.png # EDA class distribution visualization
├── training_curves.png          # Training vs validation accuracy/loss plots
├── confusion_matrix.png         # 24x24 class confusion matrix heatmap
├── requirements.txt             # Pinned project dependencies
├── .gitignore                   # Ignore large data and virtual environment
└── README.md                    # Project documentation
```

---

## 📜 License
This project is licensed under the MIT License.
