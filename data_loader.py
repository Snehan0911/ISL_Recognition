"""
data_loader.py: Dataset Preprocessing and Augmentation Pipeline for ISL Recognition
"""

import os
import tensorflow as tf

# 1. Hyperparameters & Configuration
DATASET_DIR = os.path.join("data", "extracted", "Data")
IMG_SIZE = (128, 128)      # Standardized image height and width
BATCH_SIZE = 32            # Number of images processed per gradient update
SEED = 42                  # Fixed random seed for reproducible train/val/test splits

def create_isl_datasets():
    """
    Loads images from the directory, splits them into Train, Validation, and Test,
    and applies caching and prefetching for fast GPU/CPU execution.
    """
    print(f"Loading raw dataset from '{DATASET_DIR}'...")

    # Load 80% for training + validation
    train_val_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=0.2,       # 20% reserved for validation + test
        subset="training",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical"   # One-hot encoded labels (24 classes)
    )

    # Load remaining 20%
    val_test_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=0.2,
        subset="validation",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical"
    )

    class_names = train_val_ds.class_names
    num_classes = len(class_names)

    # Split the 20% evenly into Validation (10%) and Test (10%)
    val_test_batches = tf.data.experimental.cardinality(val_test_ds).numpy()
    val_size = val_test_batches // 2

    val_ds = val_test_ds.take(val_size)
    test_ds = val_test_ds.skip(val_size)

    print("\nDataset split summary:")
    print(f"- Sign classes ({num_classes}): {class_names}")
    print(f"- Training batches      : {tf.data.experimental.cardinality(train_val_ds).numpy()}")
    print(f"- Validation batches    : {tf.data.experimental.cardinality(val_ds).numpy()}")
    print(f"- Test batches          : {tf.data.experimental.cardinality(test_ds).numpy()}")

    # Optimize pipeline performance with memory caching and background prefetching
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names

def get_augmentation_layer():
    """
    Keras Sequential data augmentation pipeline.
    Runs on the CPU/GPU seamlessly during training only.
    """
    return tf.keras.Sequential([
        tf.keras.layers.RandomRotation(0.08),       # Slight rotation (+/- 8%)
        tf.keras.layers.RandomZoom(0.08),           # Slight zoom in/out (+/- 8%)
        tf.keras.layers.RandomTranslation(0.05, 0.05) # Small shifts (+/- 5%)
    ], name="data_augmentation")

if __name__ == "__main__":
    train_ds, val_ds, test_ds, classes = create_isl_datasets()
    print("\n[SUCCESS] Dataset pipeline created and validated successfully!")

