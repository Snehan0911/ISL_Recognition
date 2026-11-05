"""
train.py: End-to-End Training and Evaluation Pipeline for ISL Recognition
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf

from data_loader import create_isl_datasets
from model import build_isl_cnn

MODEL_SAVE_PATH = "best_isl_cnn.keras"
EPOCHS = 15

def train_and_evaluate():
    print("\n" + "="*50)
    print("      STAGE 1: LOADING & SPLITTING DATASET")
    print("="*50)
    train_ds, val_ds, test_ds, class_names = create_isl_datasets()

    print("\n" + "="*50)
    print("      STAGE 2: INITIALIZING CNN MODEL")
    print("="*50)
    model = build_isl_cnn(input_shape=(128, 128, 3), num_classes=len(class_names))

    # --- Callbacks ---
    # 1. Early Stopping: Halt training when validation loss stops improving (prevents overfitting)
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True,
        verbose=1
    )

    # 2. ReduceLROnPlateau: Decays learning rate when learning stagnates
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1
    )

    # 3. ModelCheckpoint: Saves the highest validation accuracy model
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=MODEL_SAVE_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    )

    print("\n" + "="*50)
    print(f"      STAGE 3: TRAINING FOR UP TO {EPOCHS} EPOCHS")
    print("="*50)
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[early_stop, reduce_lr, checkpoint]
    )

    print("\n" + "="*50)
    print("      STAGE 4: GENERATING TRAINING CURVES")
    print("="*50)
    plot_training_curves(history)

    print("\n" + "="*50)
    print("      STAGE 5: FINAL EVALUATION ON UNSEEN TEST SET")
    print("="*50)
    evaluate_model_performance(model, test_ds, class_names)

def plot_training_curves(history):
    """Plots and saves Accuracy and Loss curves across epochs."""
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs_range = range(1, len(acc) + 1)

    plt.figure(figsize=(14, 5))

    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label="Training Accuracy", marker="o", color="blue")
    plt.plot(epochs_range, val_acc, label="Validation Accuracy", marker="s", color="orange")
    plt.title("Training vs Validation Accuracy", fontsize=12, weight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.6)

    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label="Training Loss", marker="o", color="blue")
    plt.plot(epochs_range, val_loss, label="Validation Loss", marker="s", color="red")
    plt.title("Training vs Validation Loss", fontsize=12, weight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.legend(loc="upper right")
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    curves_file = "training_curves.png"
    plt.savefig(curves_file, dpi=300)
    print(f"[SUCCESS] Training history curves saved to '{curves_file}'.")

def evaluate_model_performance(model, test_ds, class_names):
    """
    Evaluates the model on unseen test samples, generating
    Classification Report and a Confusion Matrix heatmap.
    """
    y_true_list = []
    y_pred_list = []

    print("Running inference on test dataset batches...")
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_true_list.extend(np.argmax(labels.numpy(), axis=1))
        y_pred_list.extend(np.argmax(preds, axis=1))

    y_true = np.array(y_true_list)
    y_pred = np.array(y_pred_list)

    # Classification Report
    print("\n" + "="*50)
    print("         CLASSIFICATION REPORT (TEST SET)")
    print("="*50)
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print(report)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(14, 11))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Indian Sign Language - Confusion Matrix", fontsize=14, weight="bold")
    plt.xlabel("Predicted Class", fontsize=12)
    plt.ylabel("True Class", fontsize=12)
    plt.tight_layout()

    cm_file = "confusion_matrix.png"
    plt.savefig(cm_file, dpi=300)
    print(f"[SUCCESS] Confusion matrix heatmap saved to '{cm_file}'.")
    print(f"[SUCCESS] Optimal model saved to '{MODEL_SAVE_PATH}'.")

if __name__ == "__main__":
    train_and_evaluate()

