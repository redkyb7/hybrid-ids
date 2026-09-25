import json
import os
import pickle

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

import config


def _read_dataset(
    data_path: str,
) -> pd.DataFrame:
    """
    Read the CIC collection Parquet dataset.
    """

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            "Dataset file was not found:\n"
            f"{data_path}"
        )

    return pd.read_parquet(
        data_path
    )


def _validate_dataset_schema(
    df: pd.DataFrame,
):
    """
    Validate expected target column and target classes.
    """

    if config.LABEL_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{config.LABEL_COLUMN}' "
            "was not found.\n\n"
            f"Available columns:\n"
            f"{list(df.columns)}"
        )

    labels = (
        df[config.LABEL_COLUMN]
        .astype(str)
        .str.strip()
    )

    df[config.LABEL_COLUMN] = labels

    found_classes = sorted(
        labels.unique().tolist()
    )

    expected_classes = sorted(
        config.SOURCE_CLASSES
    )

    unexpected_classes = sorted(
        set(found_classes)
        - set(expected_classes)
    )

    missing_expected_classes = sorted(
        set(expected_classes)
        - set(found_classes)
    )

    if unexpected_classes:
        message = (
            "Unexpected classes found in "
            f"'{config.LABEL_COLUMN}':\n"
            + "\n".join(
                f"- {class_name}"
                for class_name
                in unexpected_classes
            )
        )

        if config.STRICT_CLASS_VALIDATION:
            raise ValueError(
                message
                + "\n\nUpdate SOURCE_CLASSES in "
                "config.py if these are intentional."
            )

        print(
            "\nWARNING:\n"
            + message
        )

    if missing_expected_classes:
        print(
            "\nWARNING: Expected classes not present "
            "in this dataset:\n"
            + "\n".join(
                f"- {class_name}"
                for class_name
                in missing_expected_classes
            )
        )


def _remap_training_labels(labels: pd.Series) -> pd.Series:
    """Keep five named attacks and group all other source attacks together."""
    in_scope = {"Benign", *config.FOCUS_ATTACK_CLASSES}
    remapped = labels.where(labels.isin(in_scope), config.OTHER_ATTACK_CLASS)
    actual_classes = set(remapped.unique())
    if actual_classes != set(config.EXPECTED_CLASSES):
        raise ValueError(f"Unexpected remapped classes: {sorted(actual_classes)}")
    return remapped


def _get_feature_columns(
    df: pd.DataFrame,
) -> list[str]:
    """
    Select numeric feature columns and exclude target/leakage columns.
    """

    excluded_columns = set(
        config.EXCLUDE_FEATURES
        + [config.LABEL_COLUMN]
    )

    candidate_columns = [
        column
        for column in df.columns
        if column not in excluded_columns
    ]

    non_numeric_features = [
        column
        for column in candidate_columns
        if not pd.api.types.is_numeric_dtype(
            df[column]
        )
    ]

    if non_numeric_features:
        print(
            "\nIgnoring non-numeric columns:"
        )

        for column in non_numeric_features:
            print(
                f"      - {column}"
            )

    feature_columns = [
        column
        for column in candidate_columns
        if pd.api.types.is_numeric_dtype(
            df[column]
        )
    ]

    if not feature_columns:
        raise ValueError(
            "No numeric feature columns were found."
        )

    return feature_columns


def _clip_features(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Clip values using percentiles fitted only on training data.
    """

    lower_bounds = np.percentile(
        X_train,
        config.LOWER_CLIP_PERCENTILE,
        axis=0,
    ).astype(
        np.float32
    )

    upper_bounds = np.percentile(
        X_train,
        config.UPPER_CLIP_PERCENTILE,
        axis=0,
    ).astype(
        np.float32
    )

    X_train = np.clip(
        X_train,
        lower_bounds,
        upper_bounds,
    )

    X_val = np.clip(
        X_val,
        lower_bounds,
        upper_bounds,
    )

    X_test = np.clip(
        X_test,
        lower_bounds,
        upper_bounds,
    )

    return (
        X_train,
        X_val,
        X_test,
        lower_bounds,
        upper_bounds,
    )


def _print_class_distribution(
    y: np.ndarray,
    label_encoder: LabelEncoder,
    title: str,
):
    """
    Print label counts and percentages.
    """

    print(
        f"\n      {title}"
    )

    unique_classes, counts = np.unique(
        y,
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
            count / len(y)
        ) * 100

        print(
            f"        {class_name:15s}: "
            f"{count:>10,} "
            f"({percentage:6.2f}%)"
        )


def load_and_preprocess(
    data_path: str = config.DATA_PATH,
):
    """
    Load and preprocess the CIC collection Parquet dataset.

    Splits:
        80% train
        10% validation
        10% test

    Preprocessing is fitted only on training data:
    - Optional percentile clipping
    - Standard scaling
    """

    os.makedirs(config.SAVED_MODEL_DIR, exist_ok=True)

    # ========================================================
    # 1. LOAD DATA
    # ========================================================

    print("[1/6] Loading Parquet dataset...")

    df = _read_dataset(
        data_path
    )

    print(
        f"      Loaded: "
        f"{df.shape[0]:,} rows × "
        f"{df.shape[1]:,} columns"
    )

    # ========================================================
    # 2. VALIDATE TARGET / SELECT FEATURES
    # ========================================================

    print(
        "[2/6] Validating target and selecting features..."
    )

    _validate_dataset_schema(
        df
    )

    # Preserve every attack row while limiting named DL outputs to the
    # five selected classes. This happens before split and encoding.
    out_of_scope = ~df[config.LABEL_COLUMN].isin(
        ["Benign", *config.FOCUS_ATTACK_CLASSES]
    )
    print(
        f"      Mapped {int(out_of_scope.sum()):,} rows "
        f"to {config.OTHER_ATTACK_CLASS}."
    )
    df[config.LABEL_COLUMN] = _remap_training_labels(df[config.LABEL_COLUMN])

    feature_cols = _get_feature_columns(
        df
    )

    print(
        f"      Target column: "
        f"{config.LABEL_COLUMN}"
    )

    print(
        f"      Numeric features selected: "
        f"{len(feature_cols)}"
    )

    # ========================================================
    # 3. VALIDATE FEATURES / CLEAN
    # ========================================================

    print(
        "[3/6] Validating numeric features..."
    )

    X_frame = df[
        feature_cols
    ].copy()

    X_frame.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True,
    )

    valid_feature_rows = ~X_frame.isna().any(
        axis=1
    )

    valid_label_rows = (
        df[config.LABEL_COLUMN]
        .notna()
    )

    valid_rows = (
        valid_feature_rows
        & valid_label_rows
    )

    removed_invalid = int(
        (~valid_rows).sum()
    )

    if removed_invalid > 0:
        X_frame = X_frame.loc[
            valid_rows
        ].copy()

        labels = df.loc[
            valid_rows,
            config.LABEL_COLUMN,
        ].copy()
    else:
        labels = df[
            config.LABEL_COLUMN
        ].copy()

    print(
        f"      Invalid rows removed: "
        f"{removed_invalid:,}"
    )

    if config.REMOVE_DUPLICATES:
        print(
            "      Removing duplicate rows..."
        )

        before_duplicates = len(
            X_frame
        )

        duplicate_frame = X_frame.copy()
        duplicate_frame[
            config.LABEL_COLUMN
        ] = labels.to_numpy()

        duplicate_frame = duplicate_frame.drop_duplicates()

        labels = duplicate_frame.pop(
            config.LABEL_COLUMN
        )

        X_frame = duplicate_frame

        removed_duplicates = (
            before_duplicates
            - len(X_frame)
        )

        print(
            f"      Duplicate rows removed: "
            f"{removed_duplicates:,}"
        )

    # Free original dataframe before converting to NumPy.
    del df

    # ========================================================
    # 4. ENCODE LABELS
    # ========================================================

    print("[4/6] Encoding target labels...")

    X = X_frame.to_numpy(
        dtype=np.float32,
        copy=False,
    )

    y_raw = labels.to_numpy()

    del X_frame
    del labels

    label_encoder = LabelEncoder()

    y = label_encoder.fit_transform(
        y_raw
    ).astype(
        np.int32
    )

    print(
        f"      Classes: "
        f"{list(label_encoder.classes_)}"
    )

    print(
        f"      Input features: "
        f"{X.shape[1]}"
    )

    # ========================================================
    # 5. TRAIN / VALIDATION / TEST SPLIT
    # ========================================================

    print(
        "[5/6] Creating train / "
        "validation / test splits..."
    )

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

    del X
    del y

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

    _print_class_distribution(
        y_train,
        label_encoder,
        "Training class distribution:",
    )

    # ========================================================
    # 6. CLIP / SCALE / SAVE ARTIFACTS
    # ========================================================

    print(
        "\n[6/6] Scaling features and saving artifacts..."
    )

    clip_lower = None
    clip_upper = None

    if config.USE_PERCENTILE_CLIPPING:
        (
            X_train,
            X_val,
            X_test,
            clip_lower,
            clip_upper,
        ) = _clip_features(
            X_train,
            X_val,
            X_test,
        )

        print(
            "      Applied train-derived "
            "percentile clipping."
        )

    scaler = StandardScaler()

    X_train = scaler.fit_transform(
        X_train
    ).astype(
        np.float32
    )

    X_val = scaler.transform(
        X_val
    ).astype(
        np.float32
    )

    X_test = scaler.transform(
        X_test
    ).astype(
        np.float32
    )

    # Network input shape:
    # (samples, features, 1)
    X_train = X_train[
        ..., np.newaxis
    ]

    X_val = X_val[
        ..., np.newaxis
    ]

    X_test = X_test[
        ..., np.newaxis
    ]

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

    np.save(
        config.X_TEST_SAVE_PATH,
        X_test,
    )

    np.save(
        config.Y_TEST_SAVE_PATH,
        y_test,
    )

    metadata = {
        "artifact_version": "3.0.0",
        "dataset_path": os.path.abspath(
            data_path
        ),
        "label_column": config.LABEL_COLUMN,
        "excluded_columns": list(
            config.EXCLUDE_FEATURES
        ),
        "classes": label_encoder.classes_.tolist(),
        "focus_attack_classes": list(config.FOCUS_ATTACK_CLASSES),
        "other_attack_source_classes": sorted(
            set(config.SOURCE_CLASSES) - {"Benign", *config.FOCUS_ATTACK_CLASSES}
        ),
        "num_classes": int(
            len(label_encoder.classes_)
        ),
        "feature_names": feature_cols,
        "num_features": int(
            len(feature_cols)
        ),
        "model_input_shape": list(
            X_train.shape[1:]
        ),
        "percentile_clipping": {
            "enabled": bool(
                config.USE_PERCENTILE_CLIPPING
            ),
            "lower_percentile": float(
                config.LOWER_CLIP_PERCENTILE
            ),
            "upper_percentile": float(
                config.UPPER_CLIP_PERCENTILE
            ),
            "lower_bounds": (
                clip_lower.tolist()
                if clip_lower is not None
                else None
            ),
            "upper_bounds": (
                clip_upper.tolist()
                if clip_upper is not None
                else None
            ),
        },
        "split": {
            "random_state": int(
                config.RANDOM_STATE
            ),
            "train_samples": int(
                len(X_train)
            ),
            "validation_samples": int(
                len(X_val)
            ),
            "test_samples": int(
                len(X_test)
            ),
        },
    }

    with open(
        config.METADATA_SAVE_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=4,
        )

    print(
        f"      Model artifacts -> "
        f"{config.SAVED_MODEL_DIR}"
    )

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        label_encoder,
    )
