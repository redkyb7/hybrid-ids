"""
Stage 1 Machine Learning Binary Triage Filter
================================================================
Trains and benchmarks ultra-fast binary classifiers
(Random Forest vs XGBoost) on CIC collection Parquet data.

The binary target is created from ClassLabel:
    0 = Benign
    1 = Attack

This filters benign network flows before Deep Learning.
"""

import argparse
import os
import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42

np.random.seed(
    RANDOM_STATE
)

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "clean_data",
    "cic-collection.parquet",
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "updated_models",
    "experiments",
    "standalone_ml",
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True,
)

# Actual columns in cic-collection.parquet.
FINE_GRAINED_LABEL_COLUMN = "Label"
CLASS_LABEL_COLUMN = "ClassLabel"

BENIGN_LABELS = {
    "Benign",
    "BENIGN",
    "Normal",
    "Normal Traffic",
}


def load_and_prepare_data(
    data_path: str = DATA_PATH,
    sample_size: int = None,
):
    """
    Loads CIC collection Parquet flow data with stratified sampling
    across broad attack classes, creates a binary target, strips
    correlated features, and deduplicates before splitting.

    The source Parquet schema uses:
        - Label: fine-grained attack subtype
        - ClassLabel: broad category, including Benign

    ClassLabel is used to create the binary benign/attack target.
    """

    print("=" * 60)
    print(
        "1. LOADING & PREPARING CIC COLLECTION "
        "FOR STAGE 1 BINARY TRIAGE"
    )
    print("=" * 60)

    print(
        f"[*] Reading full Parquet dataset: "
        f"{data_path}"
    )

    if not os.path.exists(
        data_path
    ):
        raise FileNotFoundError(
            "Dataset file not found:\n"
            f"{data_path}"
        )

    # ========================================================
    # READ PARQUET INSTEAD OF CSV
    # ========================================================

    df_raw = pd.read_parquet(
        data_path
    )

    print(
        f"[+] Loaded raw dataset: "
        f"{df_raw.shape[0]:,} rows × "
        f"{df_raw.shape[1]} columns"
    )

    required_columns = [
        FINE_GRAINED_LABEL_COLUMN,
        CLASS_LABEL_COLUMN,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df_raw.columns
    ]

    if missing_columns:
        raise ValueError(
            "Expected label columns are missing:\n"
            + "\n".join(
                f"- {column}"
                for column in missing_columns
            )
            + "\n\nAvailable columns:\n"
            + "\n".join(
                map(
                    str,
                    df_raw.columns,
                )
            )
        )

    # Normalize broad category labels before sampling.
    df_raw[
        CLASS_LABEL_COLUMN
    ] = (
        df_raw[
            CLASS_LABEL_COLUMN
        ]
        .astype(str)
        .str.strip()
    )

    # ========================================================
    # STRATIFIED MULTI-CLASS BALANCED SAMPLING
    # ========================================================

    if (
        sample_size
        and sample_size < len(df_raw)
    ):
        print(
            "[*] Performing stratified multi-class "
            "balanced sampling "
            f"(target ~{sample_size:,} rows)..."
        )

        dfs = []

        for attack_name, group in df_raw.groupby(
            CLASS_LABEL_COLUMN
        ):
            n_group = len(
                group
            )

            if n_group <= 10_000:
                # Keep all rare attack categories.
                dfs.append(
                    group
                )
            else:
                n_sample = min(
                    n_group,
                    max(
                        15_000,
                        int(
                            sample_size
                            * (
                                n_group
                                / len(df_raw)
                            )
                        ),
                    ),
                )

                dfs.append(
                    group.sample(
                        n=n_sample,
                        random_state=RANDOM_STATE,
                    )
                )

        df = pd.concat(
            dfs,
            ignore_index=True,
        ).sample(
            frac=1.0,
            random_state=RANDOM_STATE,
        )
    else:
        df = df_raw

    # Free the original frame when sampling produced a new one.
    if df is not df_raw:
        del df_raw

    print(
        "[+] Sampled broad class distribution:\n"
        f"{df[CLASS_LABEL_COLUMN].value_counts().to_dict()}"
    )

    # ========================================================
    # CREATE BINARY TARGET
    # ========================================================

    # 0 = Benign, 1 = Attack
    y_binary = (
        ~df[CLASS_LABEL_COLUMN].isin(
            BENIGN_LABELS
        )
    ).astype(
        int
    )

    print(
        "[+] Binary class distribution:\n"
        f"{y_binary.value_counts(normalize=True).to_dict()}"
    )

    # ========================================================
    # DROP ARTIFACT / LEAKAGE COLUMNS
    # ========================================================

    # Keep this list exactly as your original implementation.
    # Any names absent from this dataset are safely ignored.
    artifact_cols = [
        "Flow Duration",
        "Flow IAT Max",
        "Flow IAT Min",
        "Flow IAT Mean",
        "Flow IAT Std",
        "Fwd IAT Total",
        "Fwd IAT Max",
        "Fwd IAT Min",
        "Fwd IAT Mean",
        "Fwd IAT Std",
        "Bwd IAT Total",
        "Bwd IAT Max",
        "Bwd IAT Min",
        "Bwd IAT Mean",
        "Bwd IAT Std",
        "Active Mean",
        "Active Max",
        "Active Min",
        "Idle Mean",
        "Idle Max",
        "Idle Min",
        "Flow Bytes/s",
        "Flow Packets/s",
        "Fwd Packets/s",
        "Bwd Packets/s",
        "Init_Win_bytes_forward",
        "Init_Win_bytes_backward",
    ]

    # IMPORTANT:
    # Drop BOTH Label and ClassLabel. Otherwise the model directly
    # sees the answer and evaluation becomes invalid.
    label_columns = [
        FINE_GRAINED_LABEL_COLUMN,
        CLASS_LABEL_COLUMN,
    ]

    X = df.drop(
        columns=label_columns + artifact_cols,
        errors="ignore",
    )

    # Your inspected dataset has numeric features, but this keeps the
    # original pipeline robust if any unexpected non-numeric column exists.
    non_numeric_columns = X.select_dtypes(
        exclude=[
            np.number,
            "bool",
        ]
    ).columns.tolist()

    if non_numeric_columns:
        print(
            "[*] Dropping non-numeric columns:\n"
            f"{non_numeric_columns}"
        )

        X = X.drop(
            columns=non_numeric_columns
        )

    X = X.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    X = X.fillna(
        0
    )

    # Free the frame which still includes labels.
    del df

    # ========================================================
    # CORRELATION PRUNING
    # ========================================================

    print(
        "[*] Performing correlation pruning "
        "(r > 0.95)..."
    )

    corr = X.corr().abs()

    upper = corr.where(
        np.triu(
            np.ones(
                corr.shape
            ),
            k=1,
        ).astype(
            bool
        )
    )

    to_drop = [
        column
        for column in upper.columns
        if any(
            upper[column] > 0.95
        )
    ]

    X = X.drop(
        columns=to_drop
    )

    print(
        f"[+] Dropped {len(to_drop)} "
        "redundant features. "
        f"Retained {X.shape[1]} robust features."
    )

    # ========================================================
    # DEDUPLICATION BEFORE SPLITTING
    # ========================================================

    print(
        "[*] Deduplicating before "
        "train/test split..."
    )

    combined = X.copy()

    combined["__label__"] = y_binary.values

    n_before = len(
        combined
    )

    combined = combined.drop_duplicates()

    n_after = len(
        combined
    )

    print(
        f"[+] Removed {n_before - n_after:,} "
        "duplicate rows "
        f"({n_before:,} -> {n_after:,})"
    )

    y = combined[
        "__label__"
    ]

    X = combined.drop(
        columns="__label__"
    )

    # ========================================================
    # STRATIFIED 70 / 15 / 15 SPLIT
    # ========================================================

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=RANDOM_STATE,
    )

    print(
        "[+] Splits -> "
        f"Train: {X_train.shape[0]:,} | "
        f"Val: {X_val.shape[0]:,} | "
        f"Test: {X_test.shape[0]:,}"
    )

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        list(X.columns),
    )


def evaluate_model(
    name,
    model,
    X_test,
    y_test,
):
    """
    Measures accuracy, precision, recall, Macro F1,
    and per-flow prediction latency in milliseconds.
    """

    # Warm-up prediction.
    _ = model.predict(
        X_test.iloc[:50]
    )

    t0 = time.perf_counter()

    preds = model.predict(
        X_test
    )

    t1 = time.perf_counter()

    latency_ms = (
        (t1 - t0)
        / len(X_test)
        * 1000
    )

    acc = accuracy_score(
        y_test,
        preds,
    )

    prec = precision_score(
        y_test,
        preds,
        zero_division=0,
    )

    rec = recall_score(
        y_test,
        preds,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        preds,
        average="macro",
    )

    print(
        f"\n{'=' * 55}\n"
        f"[EVALUATION] MODEL: {name}\n"
        f"{'=' * 55}"
    )

    print(
        f"Accuracy     : {acc:.4f}"
    )

    print(
        f"Precision    : {prec:.4f}"
    )

    print(
        f"Recall       : {rec:.4f}  "
        "(Attack capture rate)"
    )

    print(
        f"Macro F1     : {f1:.4f}  "
        "(NFR-001 >= 0.70 -> "
        f"{'PASS [OK]' if f1 >= 0.70 else 'FAIL'})"
    )

    print(
        f"Latency/Flow : {latency_ms:.5f} ms  "
        "(NFR-002 < 250ms -> "
        f"{'PASS [OK]' if latency_ms < 250 else 'FAIL'})"
    )

    print(
        "\nClassification Report:"
    )

    print(
        classification_report(
            y_test,
            preds,
            target_names=[
                "BENIGN",
                "ATTACK",
            ],
            digits=4,
            zero_division=0,
        )
    )

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
    parser = argparse.ArgumentParser(
        description=(
            "Train Stage 1 Binary "
            "ML Triage Filter."
        )
    )

    parser.add_argument(
        "--sample",
        type=int,
        default=150_000,
        help=(
            "Approximate number of rows "
            "to sample for faster training."
        ),
    )

    parser.add_argument(
        "--full",
        action="store_true",
        help=(
            "Train on the full Parquet dataset."
        ),
    )

    args = parser.parse_args()

    sample_size = (
        None
        if args.full
        else args.sample
    )

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        feature_names,
    ) = load_and_prepare_data(
        sample_size=sample_size
    )

    results = []

    # ========================================================
    # MODEL A: RANDOM FOREST
    # ========================================================

    print(
        "\n[*] Training Random Forest "
        "Classifier (Stage 1)..."
    )

    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        class_weight="balanced",
    )

    rf.fit(
        X_train,
        y_train,
    )

    results.append(
        evaluate_model(
            "Random Forest",
            rf,
            X_test,
            y_test,
        )
    )

    # ========================================================
    # MODEL B: XGBOOST
    # ========================================================

    print(
        "\n[*] Training XGBoost "
        "Classifier (Stage 1)..."
    )

    xgb_clf = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.1,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
    )

    xgb_clf.fit(
        X_train,
        y_train,
    )

    results.append(
        evaluate_model(
            "XGBoost",
            xgb_clf,
            X_test,
            y_test,
        )
    )

    # ========================================================
    # SELECT BEST MODEL
    # ========================================================

    best = max(
        results,
        key=lambda result: result[
            "f1_macro"
        ],
    )

    print("\n" + "=" * 55)

    print(
        "[+] SELECTED STAGE 1 MODEL: "
        f"{best['name']} "
        f"(F1: {best['f1_macro']:.4f})"
    )

    print("=" * 55)

    # ========================================================
    # EXPORT ARTIFACTS
    # ========================================================

    model_save_path = os.path.join(
        MODEL_DIR,
        "stage1_binary_filter.joblib",
    )

    features_save_path = os.path.join(
        MODEL_DIR,
        "stage1_feature_list.joblib",
    )

    joblib.dump(
        best["model"],
        model_save_path,
    )

    joblib.dump(
        feature_names,
        features_save_path,
    )

    print(
        f"[+] Saved Stage 1 Model   -> "
        f"{model_save_path}"
    )

    print(
        f"[+] Saved Feature Schema  -> "
        f"{features_save_path}"
    )


if __name__ == "__main__":
    main()
