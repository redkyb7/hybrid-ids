# preprocess.py

import pickle

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

import config


def load_and_preprocess(
    data_path: str = config.DATA_PATH
):
    """
    Load, clean, split, scale and prepare the network-flow dataset.

    Final split:
        80% training
        10% validation
        10% testing

    Important:
    - Scaling is fitted ONLY on training data.
    - Test data is never used during training.
    - No random oversampling is performed.
    """

    # ========================================================
    # 1. LOAD DATA
    # ========================================================

    print("[1/5] Loading dataset...")

    df = pd.read_csv(
        data_path,
        low_memory=False,
    )

    print(
        f"      Loaded: "
        f"{df.shape[0]:,} rows × {df.shape[1]} cols"
    )


    # ========================================================
    # 2. NORMALISE LABELS + CLEAN DATA
    # ========================================================

    print("[2/5] Cleaning data...")

    if config.LABEL_COLUMN not in df.columns:
        raise ValueError(
            f"Label column "
            f"'{config.LABEL_COLUMN}' "
            f"not found in dataset."
        )

    # Clean label strings
    labels = (
        df[config.LABEL_COLUMN]
        .astype(str)
        .str.strip()
    )

    # Convert known names to canonical classes
    df[config.LABEL_COLUMN] = (
        labels
        .map(config.LABEL_MAP)
        .fillna(labels)
    )

    # All columns except target are features
    feature_cols = [
        column
        for column in df.columns
        if column != config.LABEL_COLUMN
    ]

    # Convert feature columns to numeric
    df[feature_cols] = df[
        feature_cols
    ].apply(
        pd.to_numeric,
        errors="coerce",
    )

    # Replace infinity with NaN
    df.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True,
    )

    before = len(df)

    # Remove rows containing invalid values
    df.dropna(
        subset=feature_cols + [config.LABEL_COLUMN],
        inplace=True,
    )

    removed = before - len(df)

    print(
        f"      Dropped {removed:,} "
        f"rows with NaN/Inf"
    )


    # ========================================================
    # 3. FEATURES + LABEL ENCODING
    # ========================================================

    print("[3/5] Encoding labels...")

    X = df[
        feature_cols
    ].to_numpy(
        dtype=np.float32
    )

    y_raw = df[
        config.LABEL_COLUMN
    ].to_numpy()

    label_encoder = LabelEncoder()

    y = label_encoder.fit_transform(
        y_raw
    ).astype(np.int32)

    print(
        f"      Classes: "
        f"{list(label_encoder.classes_)}"
    )

    print(
        f"      Features: "
        f"{len(feature_cols)}"
    )


    # ========================================================
    # 4. TRAIN / VALIDATION / TEST SPLIT
    # ========================================================

    print(
        "[4/5] Creating train / "
        "validation / test splits..."
    )

    # 80% training
    # 20% temporary holdout
    (
        X_train,
        X_holdout,
        y_train,
        y_holdout,
    ) = train_test_split(
        X,
        y,
        test_size=config.HOLDOUT_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )

    # Split 20% holdout equally:
    # 10% validation
    # 10% test
    (
        X_val,
        X_test,
        y_val,
        y_test,
    ) = train_test_split(
        X_holdout,
        y_holdout,
        test_size=config.HOLDOUT_TEST_RATIO,
        random_state=config.RANDOM_STATE,
        stratify=y_holdout,
    )

    # Free large temporary arrays
    del X
    del y
    del X_holdout
    del y_holdout

    print(
        f"      Train      : "
        f"{len(X_train):,}"
    )

    print(
        f"      Validation : "
        f"{len(X_val):,}"
    )

    print(
        f"      Test       : "
        f"{len(X_test):,}"
    )


    # ========================================================
    # DISPLAY CLASS DISTRIBUTION
    # ========================================================

    print("\n      Training class distribution:")

    unique_classes, counts = np.unique(
        y_train,
        return_counts=True,
    )

    for class_id, count in zip(
        unique_classes,
        counts,
    ):
        class_name = label_encoder.classes_[
            class_id
        ]

        percentage = (
            count / len(y_train)
        ) * 100

        print(
            f"        "
            f"{class_name:15s}: "
            f"{count:>10,} "
            f"({percentage:6.2f}%)"
        )


    # ========================================================
    # 5. SCALE FEATURES
    # ========================================================

    print("\n[5/5] Scaling features...")

    scaler = StandardScaler()

    # Fit ONLY on training data
    X_train = scaler.fit_transform(
        X_train
    ).astype(np.float32)

    # Validation/test use training scaler
    X_val = scaler.transform(
        X_val
    ).astype(np.float32)

    X_test = scaler.transform(
        X_test
    ).astype(np.float32)


    # ========================================================
    # RESHAPE FOR 1D CNN
    #
    # Before:
    # (samples, 52)
    #
    # After:
    # (samples, 52, 1)
    # ========================================================

    X_train = X_train[
        ..., np.newaxis
    ]

    X_val = X_val[
        ..., np.newaxis
    ]

    X_test = X_test[
        ..., np.newaxis
    ]


    # ========================================================
    # SAVE PREPROCESSING ARTIFACTS
    # ========================================================

    with open(
        config.SCALER_SAVE_PATH,
        "wb",
    ) as file:
        pickle.dump(
            scaler,
            file,
        )

    with open(
        config.LABEL_ENCODER_SAVE_PATH,
        "wb",
    ) as file:
        pickle.dump(
            label_encoder,
            file,
        )

    with open(
        config.FEATURE_NAMES_SAVE_PATH,
        "wb",
    ) as file:
        pickle.dump(
            feature_cols,
            file,
        )


    # ========================================================
    # SAVE EXACT UNSEEN TEST SET
    # ========================================================

    np.save(
        config.X_TEST_SAVE_PATH,
        X_test,
    )

    np.save(
        config.Y_TEST_SAVE_PATH,
        y_test,
    )


    # ========================================================
    # ARTIFACT SUMMARY
    # ========================================================

    print(
        f"      Scaler       -> "
        f"{config.SCALER_SAVE_PATH}"
    )

    print(
        f"      Encoder      -> "
        f"{config.LABEL_ENCODER_SAVE_PATH}"
    )

    print(
        f"      Feature names -> "
        f"{config.FEATURE_NAMES_SAVE_PATH}"
    )

    print(
        f"      Test X       -> "
        f"{config.X_TEST_SAVE_PATH}"
    )

    print(
        f"      Test y       -> "
        f"{config.Y_TEST_SAVE_PATH}"
    )


    # ========================================================
    # RETURN DATA
    # ========================================================

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        label_encoder,
    )