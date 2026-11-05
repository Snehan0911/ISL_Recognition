"""
<<<<<<< HEAD
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
=======
Indian Sign Language (ISL) Recognition - Standalone Inference Script

Usage:
    python predict.py --image "path/to/image.jpg"
    python predict.py --image "data/extracted/Data/A/001.jpg" --model "best_isl_cnn.keras"
"""

import argparse
import os
import sys
import numpy as np
from PIL import Image

# 24 Indian Sign Language alphabet classes (excluding dynamic signs J and Z)
>>>>>>> cece245f4f9897901e7a5c2e6a6753f35a5c0067
CLASSES = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y'
]
<<<<<<< HEAD
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
=======

IMG_HEIGHT = 224
IMG_WIDTH = 224


def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    # Import tensorflow lazily for faster CLI help responsiveness
    import tensorflow as tf
    print(f"[INFO] Loading model from: {model_path} ...")
    model = tf.keras.models.load_model(model_path)
    print("[INFO] Model loaded successfully.")
    return model


def preprocess_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at: {image_path}")

    image = Image.open(image_path).convert("RGB")
    image_resized = image.resize((IMG_WIDTH, IMG_HEIGHT))
    image_array = np.array(image_resized, dtype=np.float32) / 255.0
    image_batch = np.expand_dims(image_array, axis=0)
    return image, image_batch


def predict(model, image_path, top_k=3):
    image, image_batch = preprocess_image(image_path)
    predictions = model.predict(image_batch, verbose=0)[0]

    # Get top-k indices
    top_indices = np.argsort(predictions)[::-1][:top_k]

    predicted_class = CLASSES[top_indices[0]]
    predicted_confidence = predictions[top_indices[0]]

    print("\n" + "=" * 45)
    print("      ISL RECOGNITION PREDICTION RESULT      ")
    print("=" * 45)
    print(f"Image File       : {image_path}")
    print(f"Predicted Sign   : {predicted_class}")
    print(f"Confidence       : {predicted_confidence * 100:.2f}%")
    print("-" * 45)
    print(f"Top {top_k} Predictions:")
    for rank, idx in enumerate(top_indices, 1):
        print(f"  {rank}. Sign '{CLASSES[idx]}': {predictions[idx] * 100:.2f}%")
    print("=" * 45 + "\n")

    return predicted_class, predicted_confidence


def main():
    parser = argparse.ArgumentParser(
        description="Predict Indian Sign Language (ISL) alphabet from an image."
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the hand gesture image (.jpg, .jpeg, .png)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="best_isl_cnn.keras",
        help="Path to the trained .keras model (default: best_isl_cnn.keras)"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=3,
        help="Number of top predictions to display (default: 3)"
    )

    args = parser.parse_args()

    try:
        model = load_model(args.model)
        predict(model, args.image, top_k=args.top_k)
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
>>>>>>> cece245f4f9897901e7a5c2e6a6753f35a5c0067
