"""
SentinelFlow IDS - Stage 1 Binary Triage Model Training
========================================================
Trains a high-throughput, port-invariant Random Forest binary filter on CICIDS2017.
Filters >80-99% of benign network traffic in < 35ms before escalating suspicious flows to Stage 2.

Output Artifacts:
  - backend/models/stage1_binary_filter.joblib
  - backend/models/stage1_feature_list.joblib
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42

# Resolve paths relative to project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DATA_PATH = os.path.join(PROJECT_ROOT, "clean_data", "cicids2017_cleaned.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "backend", "models")


def train_stage1():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"[*] Loading dataset: {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()

    # Define features to exclude (prevent data leakage and artifact memorization)
    EXCLUDE_COLS = {
        'Label', 'Attack Type', 'Source IP', 'Destination IP', 
        'Flow ID', 'Timestamp', 'External IP',
        'Destination Port', 'Init_Win_bytes_forward', 
        'Init_Win_bytes_backward', 'Flow Duration'
    }

    numeric_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns 
        if c not in EXCLUDE_COLS
    ]
    print(f"[+] Selected {len(numeric_cols)} behavioral features (Port & Window invariant).")

    # Binary label: 0 for Benign, 1 for Attack
    y_binary = (df['Attack Type'] != 'Normal Traffic').astype(int)
    X = df[numeric_cols].copy()

    # Clean non-finite values
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    print(f"[*] Total dataset size: {len(X):,} flows")
    print(f"[*] Class distribution: Benign={sum(y_binary==0):,}, Malicious={sum(y_binary==1):,}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_binary, test_size=0.20, random_state=RANDOM_STATE, stratify=y_binary
    )

    print("[*] Training Stage 1 Balanced Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        class_weight='balanced_subsample',
        n_jobs=-1,
        random_state=RANDOM_STATE
    )
    rf.fit(X_train, y_train)

    # Evaluation
    y_pred_proba = rf.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.10).astype(int)

    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"\n[+] Holdout ROC-AUC Score: {roc_auc:.5f}")
    print("\n[+] Classification Report (Threshold tau = 0.10):")
    print(classification_report(y_test, y_pred, target_names=['Benign (0)', 'Malicious (1)'], digits=4))

    # Save artifacts
    model_file = os.path.join(OUTPUT_DIR, "stage1_binary_filter.joblib")
    feat_file = os.path.join(OUTPUT_DIR, "stage1_feature_list.joblib")
    
    joblib.dump(rf, model_file)
    joblib.dump(numeric_cols, feat_file)
    print(f"[+] Saved model to: {model_file}")
    print(f"[+] Saved feature list ({len(numeric_cols)} features) to: {feat_file}")


if __name__ == "__main__":
    train_stage1()
