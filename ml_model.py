"""
SentinelFlow IDS - Stage 1 Machine Learning Binary Triage Filter
================================================================
Trains and benchmarks ultra-fast binary classifiers (Random Forest vs XGBoost)
on CICIDS2017 to filter >80% benign network flows in < 5ms before Deep Learning.
"""

import argparse
import os
import time
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
import xgboost as xgb

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "clean_data", "cicids2017_cleaned.csv")
MODEL_DIR = os.path.join(BASE_DIR, "backend", "models")
os.makedirs(MODEL_DIR, exist_ok=True)


def load_and_prepare_data(data_path: str = DATA_PATH, sample_size: int = None):
    """
    Loads CICIDS2017 flow data, creates binary target, strips correlated features,
    and deduplicates before split to prevent data leakage.
    """
    print("=" * 60)
    print("1. LOADING & PREPARING CICIDS2017 FOR STAGE 1 BINARY TRIAGE")
    print("=" * 60)
    print(f"[*] Reading dataset: {data_path}")

    if sample_size:
        print(f"[*] Subsampling {sample_size:,} rows for rapid training...")
        # Read header first
        df_sample = pd.read_csv(data_path, nrows=sample_size * 2, low_memory=False)
        if len(df_sample) > sample_size:
            df = df_sample.sample(n=sample_size, random_state=RANDOM_STATE)
        else:
            df = df_sample
    else:
        df = pd.read_csv(data_path, low_memory=False)

    print(f"[+] Loaded raw shape: {df.shape}")

    # Create Binary Label: 0 = Benign/Normal, 1 = Attack
    normal_labels = {"Normal Traffic", "BENIGN", "Normal"}
    y_binary = (~df["Attack Type"].isin(normal_labels)).astype(int)
    print(f"[+] Binary class distribution:\n{y_binary.value_counts(normalize=True).to_dict()}")

    # Feature Matrix X
    X = df.drop(columns=["Attack Type"], errors="ignore")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))

    # Feature Selection: Drop Collinear Features (r > 0.95)
    print("[*] Performing correlation pruning (r > 0.95)...")
    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [c for c in upper.columns if any(upper[c] > 0.95)]
    X = X.drop(columns=to_drop)
    print(f"[+] Dropped {len(to_drop)} redundant features. Retained {X.shape[1]} features.")

    # Deduplicate BEFORE splitting to prevent train/test leakage
    print("[*] Deduplicating before train/test split...")
    combined = X.copy()
    combined["__label__"] = y_binary.values
    n_before = len(combined)
    combined = combined.drop_duplicates()
    n_after = len(combined)
    print(f"[+] Removed {n_before - n_after:,} duplicate rows ({n_before:,} -> {n_after:,})")

    y = combined["__label__"]
    X = combined.drop(columns="__label__")

    # Stratified Split: 70% Train, 15% Val, 15% Test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_STATE
    )
    print(f"[+] Splits -> Train: {X_train.shape[0]:,} | Val: {X_val.shape[0]:,} | Test: {X_test.shape[0]:,}")

    return X_train, X_val, X_test, y_train, y_val, y_test, list(X.columns)


def evaluate_model(name, model, X_test, y_test):
    """Measures accuracy, precision, recall, Macro F1, and per-flow latency in ms."""
    _ = model.predict(X_test.iloc[:50])  # Warmup
    t0 = time.perf_counter()
    preds = model.predict(X_test)
    t1 = time.perf_counter()
    latency_ms = (t1 - t0) / len(X_test) * 1000

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, average="macro")

    print(f"\n{'='*55}\n[EVALUATION] MODEL: {name}\n{'='*55}")
    print(f"Accuracy     : {acc:.4f}")
    print(f"Precision    : {prec:.4f}")
    print(f"Recall       : {rec:.4f}  (Attack capture rate)")
    print(f"Macro F1     : {f1:.4f}  (NFR-001 >= 0.70 -> {'PASS [OK]' if f1 >= 0.70 else 'FAIL'})")
    print(f"Latency/Flow : {latency_ms:.5f} ms  (NFR-002 < 250ms -> {'PASS [OK]' if latency_ms < 250 else 'FAIL'})")
    print("\nClassification Report:")
    print(classification_report(y_test, preds, target_names=["BENIGN", "ATTACK"], digits=4))

    return {
        "name": name,
        "model": model,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_macro": f1,
        "latency_ms": latency_ms,
        "preds": preds,
    }


def main():
    parser = argparse.ArgumentParser(description="Train Stage 1 Binary ML Triage Filter.")
    parser.add_argument("--sample", type=int, default=150000, help="Number of rows to sample for fast training")
    parser.add_argument("--full", action="store_true", help="Train on full 2.5M rows dataset")
    args = parser.parse_args()

    sample_size = None if args.full else args.sample
    X_train, X_val, X_test, y_train, y_val, y_test, feature_names = load_and_prepare_data(sample_size=sample_size)

    results = []

    # Model A: Random Forest Binary Classifier
    print("\n[*] Training Random Forest Classifier (Stage 1)...")
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        class_weight="balanced",
    )
    rf.fit(X_train, y_train)
    results.append(evaluate_model("Random Forest", rf, X_test, y_test))

    # Model B: XGBoost Binary Classifier
    print("\n[*] Training XGBoost Classifier (Stage 1)...")
    xgb_clf = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.1,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
    )
    xgb_clf.fit(X_train, y_train)
    results.append(evaluate_model("XGBoost", xgb_clf, X_test, y_test))

    # Select Best Model
    best = max(results, key=lambda r: r["f1_macro"])
    print("\n" + "=" * 55)
    print(f"[+] SELECTED STAGE 1 MODEL: {best['name']} (F1: {best['f1_macro']:.4f})")
    print("=" * 55)

    # Export Stage 1 Artifacts
    model_save_path = os.path.join(MODEL_DIR, "stage1_binary_filter.joblib")
    features_save_path = os.path.join(MODEL_DIR, "stage1_feature_list.joblib")

    joblib.dump(best["model"], model_save_path)
    joblib.dump(feature_names, features_save_path)

    print(f"[+] Saved Stage 1 Model   -> {model_save_path}")
    print(f"[+] Saved Feature Schema  -> {features_save_path}")


if __name__ == "__main__":
    main()
