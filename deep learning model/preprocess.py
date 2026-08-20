# preprocess.py
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import RandomOverSampler

import config


def load_and_preprocess(data_path: str = config.DATA_PATH):
    print("[1/5] Loading dataset...")
    df = pd.read_csv(data_path)

    print(f"      Loaded: {df.shape[0]:,} rows × {df.shape[1]} cols")

    # ── Normalise label names ──────────────────────────────────────────────
    df[config.LABEL_COLUMN] = (
        df[config.LABEL_COLUMN]
        .str.strip()
        .map(config.LABEL_MAP)
        .fillna(df[config.LABEL_COLUMN].str.strip())  # keep unmapped as-is
    )

    print("[2/5] Cleaning data...")

    # Drop non-numeric feature columns (keep label)
    feature_cols = [c for c in df.columns if c != config.LABEL_COLUMN]
    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce")

    # Replace inf with NaN, then drop rows with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    before = len(df)
    df.dropna(inplace=True)
    print(f"      Dropped {before - len(df):,} rows with NaN/Inf")

    # ── Features & labels ─────────────────────────────────────────────────
    X = df[feature_cols].values.astype(np.float32)
    y_raw = df[config.LABEL_COLUMN].values

    print("[3/5] Encoding labels...")
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    print(f"      Classes: {list(le.classes_)}")

    # ── Train / test split ────────────────────────────────────────────────
    print("[4/5] Splitting & scaling...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # ── Handle class imbalance (ratio 1075:1 is severe) ───────────────────
    print("[5/5] Resampling training set to handle imbalance...")
    ros = RandomOverSampler(random_state=config.RANDOM_STATE)
    X_train, y_train = ros.fit_resample(X_train, y_train)
    print(f"      After resampling: {X_train.shape[0]:,} training samples")

    # ── Reshape for CNN: (samples, features, 1) ───────────────────────────
    X_train = X_train[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    # ── Persist scaler & encoder for inference ────────────────────────────
    with open(config.SCALER_SAVE_PATH, "wb") as f:
        pickle.dump(scaler, f)
    with open(config.LABEL_ENCODER_SAVE_PATH, "wb") as f:
        pickle.dump(le, f)
    print(f"      Scaler  -> {config.SCALER_SAVE_PATH}")
    print(f"      Encoder -> {config.LABEL_ENCODER_SAVE_PATH}")

    return X_train, X_test, y_train, y_test, le