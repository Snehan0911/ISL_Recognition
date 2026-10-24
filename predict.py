"""
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
CLASSES = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y'
]

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
