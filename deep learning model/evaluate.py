# evaluate.py
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf

import config
from train import train


def evaluate():
    model, history, X_test, y_test, le = train()

    print("\n========== EVALUATION ==========")
    y_pred = np.argmax(model.predict(X_test, batch_size=config.BATCH_SIZE), axis=1)

    print("\nClassification Report:")
    print(
        classification_report(
            y_test, y_pred,
            target_names=le.classes_,
            digits=4,
        )
    )

    # ── Confusion matrix ──────────────────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=le.classes_,
        yticklabels=le.classes_,
        cmap="Blues",
    )
    plt.title("Confusion Matrix — CNN NIDS")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    print("Confusion matrix saved → confusion_matrix.png")

    # ── Training curves ───────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(history.history["accuracy"], label="Train Acc")
    axes[0].plot(history.history["val_accuracy"], label="Val Acc")
    axes[0].set_title("Accuracy")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="Train Loss")
    axes[1].plot(history.history["val_loss"], label="Val Loss")
    axes[1].set_title("Loss")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150)
    print("Training curves saved → training_curves.png")


if __name__ == "__main__":
    evaluate()