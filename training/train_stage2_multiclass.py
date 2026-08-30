"""
SentinelFlow IDS - Stage 2 Multi-Class Attack Categorisation Model Training
===========================================================================
Trains a balanced multi-class Random Forest classifier on 425k+ CICIDS2017 attack flows.
Accurately categorises escalated threats into: DoS, DDoS, Port Scan, Brute Force, Web Attack, and Botnet.

Output Artifacts:
  - backend/models/stage2_attack_classifier.joblib
  - backend/models/stage2_attack_classes.joblib
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42

# Resolve paths relative to project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DATA_PATH = os.path.join(PROJECT_ROOT, "clean_data", "cicids2017_cleaned.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "backend", "models")


def train_stage2():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"[*] Loading dataset: {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()

    # Load the exact 48 behavioral features used in Stage 1
    feature_list_path = os.path.join(OUTPUT_DIR, "stage1_feature_list.joblib")
    if os.path.exists(feature_list_path):
        feature_cols = joblib.load(feature_list_path)
        print(f"[+] Loaded {len(feature_cols)} Stage 1 behavioral features.")
    else:
        EXCLUDE_COLS = {
            'Label', 'Attack Type', 'Source IP', 'Destination IP', 
            'Flow ID', 'Timestamp', 'External IP',
            'Destination Port', 'Init_Win_bytes_forward', 
            'Init_Win_bytes_backward', 'Flow Duration'
        }
        feature_cols = [
            c for c in df.select_dtypes(include=[np.number]).columns 
            if c not in EXCLUDE_COLS
        ]

    # Filter to attack flows only (Benign traffic handled by Stage 1 triage)
    attack_df = df[df['Attack Type'] != 'Normal Traffic'].copy()
    print(f"[*] Total attack flows: {len(attack_df):,}")
    print("[*] Attack Breakdown in Dataset:")
    print(attack_df['Attack Type'].value_counts())

    X = attack_df[feature_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = attack_df['Attack Type']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    print("\n[*] Training Stage 2 Multi-Class Balanced Random Forest...")
    rf_stage2 = RandomForestClassifier(
        n_estimators=100,
        max_depth=22,
        class_weight='balanced_subsample',
        n_jobs=-1,
        random_state=RANDOM_STATE
    )
    rf_stage2.fit(X_train, y_train)

    y_pred = rf_stage2.predict(X_test)
    print("\n[+] Stage 2 Holdout Evaluation (85,139 unseen attack flows):")
    print(classification_report(y_test, y_pred, digits=4))

    # Save Stage 2 artifacts
    model_file = os.path.join(OUTPUT_DIR, "stage2_attack_classifier.joblib")
    classes_file = os.path.join(OUTPUT_DIR, "stage2_attack_classes.joblib")

    joblib.dump(rf_stage2, model_file)
    joblib.dump(list(rf_stage2.classes_), classes_file)
    print(f"[+] Saved model to: {model_file}")
    print(f"[+] Saved class list {list(rf_stage2.classes_)} to: {classes_file}")


if __name__ == "__main__":
    train_stage2()
