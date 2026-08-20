# evaluate.py
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import tensorflow as tf

import config


def evaluate(sample_size: int = 100000):
    print("=" * 60)
    print("🎯 EVALUATING PRE-TRAINED DEEP LEARNING MODEL (1D-CNN)")
    print("=" * 60)

    # 1. Load Pre-trained Model & Preprocessing Artifacts
    if not os.path.exists(config.MODEL_SAVE_PATH):
        raise FileNotFoundError(f"Model file not found at: {config.MODEL_SAVE_PATH}. Please train the model first.")

    print(f"[*] Loading pre-trained model: {config.MODEL_SAVE_PATH}")
    model = tf.keras.models.load_model(config.MODEL_SAVE_PATH)

    with open(config.SCALER_SAVE_PATH, "rb") as f:
        scaler = pickle.load(f)
    with open(config.LABEL_ENCODER_SAVE_PATH, "rb") as f:
        le = pickle.load(f)

    classes = list(le.classes_)
    print(f"[+] Loaded Model & Artifacts successfully! Classes ({len(classes)}): {classes}")

    # 2. Load Evaluation / Test Dataset
    print(f"\n[*] Loading evaluation dataset from: {config.DATA_PATH}")
    if sample_size:
        print(f"[*] Subsampling {sample_size:,} flows for fast evaluation...")
        df = pd.read_csv(config.DATA_PATH, nrows=sample_size * 2, low_memory=False)
        if len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=config.RANDOM_STATE)
    else:
        df = pd.read_csv(config.DATA_PATH, low_memory=False)

    # Map labels to canonical names
    df[config.LABEL_COLUMN] = (
        df[config.LABEL_COLUMN]
        .str.strip()
        .map(config.LABEL_MAP)
        .fillna(df[config.LABEL_COLUMN].str.strip())
    )

    # Filter to only known classes
    df = df[df[config.LABEL_COLUMN].isin(classes)]

    # Clean numeric features
    feature_cols = [c for c in df.columns if c != config.LABEL_COLUMN]
    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    X_raw = df[feature_cols].values.astype(np.float32)
    y_raw = df[config.LABEL_COLUMN].values

    # Transform features and labels
    X_scaled = scaler.transform(X_raw)
    X_test = X_scaled[..., np.newaxis]  # (N, 52, 1)
    y_test = le.transform(y_raw)

    print(f"[+] Evaluated samples: {len(y_test):,}")

    # 3. Model Inference
    print("\n[*] Running inference on evaluation data...")
    raw_preds = model.predict(X_test, batch_size=2048, verbose=1)
    y_pred = np.argmax(raw_preds, axis=1)

    # 4. Metrics & Classification Report
    print("\n" + "=" * 60)
    print("📊 CLASSIFICATION REPORT (NFR-001 BENCHMARK)")
    print("=" * 60)
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=classes,
            digits=4,
            zero_division=0,
        )
    )

    # 5. Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=classes,
        yticklabels=classes,
        cmap="Blues",
    )
    plt.title("Confusion Matrix - SentinelFlow 1D-CNN NIDS")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    cm_output_path = os.path.join(os.path.dirname(__file__), "confusion_matrix.png")
    plt.savefig(cm_output_path, dpi=150)
    print(f"[+] Confusion matrix plot saved -> {cm_output_path}")
    print("=" * 60)


if __name__ == "__main__":
    evaluate()