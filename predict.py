"""
predict.py: Standalone Inference Script for Indian Sign Language (ISL) Recognition

Usage:
    python predict.py --image "data/extracted/Data/A/001.jpg"
    python predict.py --image "path/to/any_sign.jpg" --top_k 3
"""

import os
import argparse
import numpy as np
from PIL import Image
import tensorflow as tf

# The 24 ISL sign alphabet classes (excluding dynamic motion signs J and Z)
CLASSES = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y'
]
IMG_SIZE = (128, 128)
DEFAULT_MODEL_PATH = "best_isl_cnn.keras"

def load_isl_model(model_path=DEFAULT_MODEL_PATH):
    """Loads the trained Keras model."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at '{model_path}'. Please run train.py first.")
    return tf.keras.models.load_model(model_path)

def preprocess_image(image_path):
    """
    Loads and resizes image to (128, 128, 3).
    Note: Pixel rescaling is handled by the model's internal Rescaling layer.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at '{image_path}'.")
    
    img = Image.open(image_path).convert("RGB")
    img_resized = img.resize(IMG_SIZE)
    img_array = np.array(img_resized, dtype=np.float32)
    # Add batch dimension: (1, 128, 128, 3)
    img_batch = np.expand_dims(img_array, axis=0)
    return img, img_batch

def predict_sign(image_path, model_path=DEFAULT_MODEL_PATH, top_k=3):
    """Predicts the sign for a given image."""
    model = load_isl_model(model_path)
    img, img_batch = preprocess_image(image_path)

    # Perform inference
    predictions = model.predict(img_batch, verbose=0)[0]
    
    # Get top-k indices sorted by confidence
    top_indices = np.argsort(predictions)[::-1][:top_k]

    predicted_class = CLASSES[top_indices[0]]
    predicted_confidence = predictions[top_indices[0]] * 100

    print("\n" + "="*50)
    print("      INDIAN SIGN LANGUAGE PREDICTION RESULT")
    print("="*50)
    print(f"Target Image         : {image_path}")
    print(f"Predicted Sign Class : {predicted_class}")
    print(f"Confidence           : {predicted_confidence:.2f}%")
    print("\nTop Candidates:")
    for rank, idx in enumerate(top_indices, 1):
        print(f"  {rank}. Sign '{CLASSES[idx]}': {predictions[idx] * 100:.2f}%")
    print("="*50)

    return predicted_class, predicted_confidence

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict Indian Sign Language from an image")
    parser.add_argument("--image", type=str, default="data/extracted/Data/A/001.jpg", help="Path to input image")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH, help="Path to trained .keras model")
    parser.add_argument("--top_k", type=int, default=3, help="Number of top predictions to display")

    args = parser.parse_args()
    predict_sign(args.image, args.model, args.top_k)
