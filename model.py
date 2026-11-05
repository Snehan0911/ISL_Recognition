"""
model.py: Custom Deep CNN Architecture for Indian Sign Language (ISL) Recognition
"""

import tensorflow as tf
from tensorflow.keras import layers, models

def build_isl_cnn(input_shape=(128, 128, 3), num_classes=24):
    """
    Constructs a modern 4-block Convolutional Neural Network with
    Batch Normalization and Dropout for robust sign classification.
    """
    model = models.Sequential([
        # Input Layer
        layers.Input(shape=input_shape),

        # Normalization layer: rescales [0, 255] to [0.0, 1.0]
        layers.Rescaling(1.0 / 255.0, name="pixel_rescaling"),

        # --- Conv Block 1: Basic Edges and Gradients ---
        layers.Conv2D(32, (3, 3), padding="same", name="conv1"),
        layers.BatchNormalization(name="bn1"),
        layers.Activation("relu", name="relu1"),
        layers.MaxPooling2D(pool_size=(2, 2), name="pool1"),

        # --- Conv Block 2: Finger contours & Curves ---
        layers.Conv2D(64, (3, 3), padding="same", name="conv2"),
        layers.BatchNormalization(name="bn2"),
        layers.Activation("relu", name="relu2"),
        layers.MaxPooling2D(pool_size=(2, 2), name="pool2"),

        # --- Conv Block 3: Hand shapes & Part configurations ---
        layers.Conv2D(128, (3, 3), padding="same", name="conv3"),
        layers.BatchNormalization(name="bn3"),
        layers.Activation("relu", name="relu3"),
        layers.MaxPooling2D(pool_size=(2, 2), name="pool3"),

        # --- Conv Block 4: Complex Gesture Semantics ---
        layers.Conv2D(256, (3, 3), padding="same", name="conv4"),
        layers.BatchNormalization(name="bn4"),
        layers.Activation("relu", name="relu4"),
        layers.MaxPooling2D(pool_size=(2, 2), name="pool4"),

        # --- Classification Head ---
        # Global Average Pooling reduces spatial dimensions to a 1D vector
        layers.GlobalAveragePooling2D(name="gap"),
        
        # Fully Connected Layer
        layers.Dense(256, activation="relu", name="dense1"),
        layers.BatchNormalization(name="bn_dense"),
        layers.Dropout(0.4, name="dropout_regularization"),

        # Final Output Layer: 24 Sign Classes with Softmax probabilities
        layers.Dense(num_classes, activation="softmax", name="output_probabilities")
    ], name="ISL_Recognition_CNN")

    # Compile model with Adam optimizer and Categorical Crossentropy loss
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model

if __name__ == "__main__":
    model = build_isl_cnn()
    print("\n" + "="*50)
    print("      ISL RECOGNITION CNN ARCHITECTURE")
    print("="*50)
    model.summary()

