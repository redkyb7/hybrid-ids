"""Retrain both IDS stages on one deduplicated, untouched holdout.

Run in Colab from the repository root after installing the DL training
requirements plus XGBoost and joblib. Output is a new bundle; uploaded model
files are left untouched.
"""

import argparse
import gc
import hashlib
import importlib
import json
import pickle
import platform
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
dl_config = importlib.import_module("deep learning model.config")
build_nids_model = importlib.import_module(
    "deep learning model.model"
).build_nids_model

ML_EXCLUDE = {
    "Flow Duration", "Flow IAT Max", "Flow IAT Min", "Flow IAT Mean",
    "Flow IAT Std", "Fwd IAT Total", "Fwd IAT Max", "Fwd IAT Min",
    "Fwd IAT Mean", "Fwd IAT Std", "Bwd IAT Total", "Bwd IAT Max",
    "Bwd IAT Min", "Bwd IAT Mean", "Bwd IAT Std", "Active Mean",
    "Active Max", "Active Min", "Idle Mean", "Idle Max", "Idle Min",
    "Flow Bytes/s", "Flow Packets/s", "Fwd Packets/s", "Bwd Packets/s",
    "Init_Win_bytes_forward", "Init_Win_bytes_backward",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_dataset_batches(data_path: Path, feature_names: list[str],
                         batch_rows: int):
    """Read the large Parquet row group without materializing every row."""
    import pyarrow.parquet as parquet

    source = parquet.ParquetFile(data_path)
    columns = feature_names + ["ClassLabel"]
    for batch in source.iter_batches(batch_size=batch_rows, columns=columns):
        yield batch.to_pandas()


def prepare_shared_data(data_path: Path, batch_rows: int = 50_000):
    with (ROOT / "updated_models" / "extracted" / "dl" /
          "metadata.json").open(encoding="utf-8") as source:
        reference = json.load(source)
    feature_names = reference["feature_names"]
    expected_classes = reference["classes"]
    candidates = [name for name in feature_names if name not in ML_EXCLUDE]
    candidate_columns = [feature_names.index(name) for name in candidates]

    print("Reading and deduplicating Parquet batches...", flush=True)
    # Deduplicate using the fixed ML candidate signature. The 57-feature
    # vectors are almost all unique, while the ML-visible behavior repeats.
    # Keep one seeded random full DL vector per unambiguous signature so
    # source-file ordering does not decide the representative.
    observed: dict[bytes, tuple[str, int, bytes, float]] = {}
    conflicts: dict[bytes, set[str]] = {}
    rng = np.random.default_rng(42)
    raw_rows = 0
    invalid_rows = 0
    for frame in iter_dataset_batches(data_path, feature_names, batch_rows):
        values = frame[feature_names].to_numpy(dtype=np.float32, copy=True)
        labels = frame["ClassLabel"].to_numpy()
        valid = np.isfinite(values).all(axis=1) & pd.notna(labels)
        invalid_rows += int(np.count_nonzero(~valid))
        signatures = values[:, candidate_columns]
        priorities = rng.random(len(frame))
        for local_index in np.flatnonzero(valid):
            key = signatures[local_index].tobytes()
            label = str(labels[local_index]).strip()
            previous = observed.get(key)
            if previous is None:
                observed[key] = (
                    label, raw_rows + int(local_index),
                    values[local_index].tobytes(),
                    float(priorities[local_index]),
                )
            elif previous[0] != label:
                conflicts.setdefault(key, {previous[0]}).add(label)
            elif priorities[local_index] > previous[3]:
                observed[key] = (
                    label, raw_rows + int(local_index),
                    values[local_index].tobytes(),
                    float(priorities[local_index]),
                )
        raw_rows += len(frame)
        if raw_rows % 1_000_000 < len(frame):
            print(f"Read {raw_rows:,} rows; distinct vectors {len(observed):,}",
                  flush=True)

    after_duplicates = len(observed) + sum(
        len(labels) - 1 for labels in conflicts.values())
    conflicting_count = sum(len(labels) for labels in conflicts.values())
    retained = len(observed) - len(conflicts)
    X = np.empty((retained, len(feature_names)), dtype=np.float32)
    row_ids = np.empty(retained, dtype=np.int64)
    raw_labels = []
    index = 0
    for key, (label, original_row_id, full_vector, _priority) in observed.items():
        if key in conflicts:
            continue
        X[index] = np.frombuffer(full_vector, dtype=np.float32)
        row_ids[index] = original_row_id
        raw_labels.append(label)
        index += 1
    del observed, conflicts
    gc.collect()
    print(f"Rows: raw={raw_rows:,}, invalid={invalid_rows:,}, "
          f"after duplicates={after_duplicates:,}, "
          f"ambiguous removed={conflicting_count:,}, retained={retained:,}",
          flush=True)

    if sorted(set(raw_labels)) != sorted(expected_classes):
        raise ValueError("The dataset classes differ from the uploaded schema")
    encoder = LabelEncoder()
    y = encoder.fit_transform(raw_labels).astype(np.int32)
    if list(encoder.classes_) != sorted(expected_classes):
        raise ValueError("Label encoder class order changed")
    if encoder.classes_[0] != "Benign":
        raise ValueError("Stage 1 training expects Benign to have class index 0")
    del raw_labels
    positions = np.arange(retained)
    train_pos, holdout_pos = train_test_split(
        positions, test_size=0.30, stratify=y, random_state=42)
    val_pos, test_pos = train_test_split(
        holdout_pos, test_size=0.50, stratify=y[holdout_pos], random_state=42)

    train_frame = pd.DataFrame(
        X[np.ix_(train_pos, candidate_columns)], columns=candidates)
    train_corr = train_frame.corr().abs()
    del train_frame
    upper = train_corr.where(np.triu(np.ones(train_corr.shape), k=1).astype(bool))
    ml_features = [name for name in candidates
                   if not (upper[name] > 0.95).any()]
    if not ml_features:
        raise ValueError("Correlation pruning removed every ML feature")
    print(f"ML features selected on training rows: {len(ml_features)}", flush=True)

    gc.collect()
    return (X, y, encoder, feature_names, ml_features, row_ids,
            train_pos, val_pos, test_pos,
            {"raw_rows": raw_rows,
             "after_signature_deduplication": after_duplicates,
             "ambiguous_removed": conflicting_count,
             "invalid_rows": invalid_rows,
             "dedup_signature_columns": candidates,
             "representative_policy": "highest seeded random priority",
             "representative_random_state": 42})


def stage1_frames(X, positions, feature_names, ml_features):
    columns = [feature_names.index(name) for name in ml_features]
    return pd.DataFrame(X[np.ix_(positions, columns)], columns=ml_features)


def fit_stage1(X, y, feature_names, ml_features, train_pos):
    train_frame = stage1_frames(X, train_pos, feature_names, ml_features)
    target = (y[train_pos] != 0).astype(np.int8)
    candidates = {
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=20, class_weight="balanced",
            n_jobs=-1, random_state=42),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=150, max_depth=6, learning_rate=0.1,
            tree_method="hist", n_jobs=-1, random_state=42,
            eval_metric="logloss"),
    }
    for name, model in candidates.items():
        print(f"Training Stage 1: {name}", flush=True)
        model.fit(train_frame, target)
    del train_frame
    return candidates


def scale_dl(X, train_pos, val_pos, test_pos):
    percentiles = np.percentile(X[train_pos], [0.5, 99.5], axis=0)
    lower, upper = percentiles.astype(np.float32)
    scaler = StandardScaler()
    train_clipped = np.clip(X[train_pos], lower, upper)
    scaler.fit(train_clipped)

    def transform(values):
        clipped = np.clip(values, lower, upper)
        return scaler.transform(clipped).astype(np.float32)[..., np.newaxis]

    X_train = transform(train_clipped)
    del train_clipped
    X_val = transform(X[val_pos])
    X_test = transform(X[test_pos])
    return X_train, X_val, X_test, scaler, lower, upper


def fit_stage2(X_train, X_val, y_train, y_val, classes, epochs, batch_size):
    tf.keras.utils.set_random_seed(42)
    model = build_nids_model((X_train.shape[1], 1), len(classes))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=dl_config.LEARNING_RATE),
        loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    counts = np.bincount(y_train, minlength=len(classes))
    balanced = len(y_train) / (len(classes) * counts)
    softened = np.clip(np.sqrt(balanced), 0.25, 10.0)
    class_weight = {index: float(weight)
                    for index, weight in enumerate(softened)}
    model.fit(
        X_train, y_train, validation_data=(X_val, y_val),
        batch_size=batch_size, epochs=epochs, shuffle=True,
        class_weight=class_weight,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=6, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
        ], verbose=2)
    return model


def combined_attack(gate_probability, dl_probabilities, gate_threshold,
                    dl_threshold, benign_index):
    predicted = np.argmax(dl_probabilities, axis=1)
    confidence = np.max(dl_probabilities, axis=1)
    gated = gate_probability >= gate_threshold
    attack = gated & ((confidence < dl_threshold) |
                      (predicted != benign_index))
    return attack, gated, predicted, confidence


def choose_policy(models, X, y, feature_names, ml_features, val_pos,
                  dl_val_probabilities, benign_index):
    frame = stage1_frames(X, val_pos, feature_names, ml_features)
    target = y[val_pos] != benign_index
    best = None
    for name, model in models.items():
        gate_probability = model.predict_proba(frame)[:, 1]
        for gate_threshold in np.linspace(0.05, 0.95, 19):
            for dl_threshold in (0.50, 0.60, 0.70, 0.80, 0.90):
                attack, _, _, _ = combined_attack(
                    gate_probability, dl_val_probabilities, gate_threshold,
                    dl_threshold, benign_index)
                score = float(f1_score(target, attack, zero_division=0))
                recall = float(recall_score(target, attack, zero_division=0))
                candidate = {
                    "model_name": name,
                    "stage1_threshold": float(gate_threshold),
                    "dl_threshold": float(dl_threshold),
                    "validation_attack_f1": score,
                    "validation_attack_recall": recall,
                }
                if best is None or (score, recall) > (
                    best["validation_attack_f1"],
                    best["validation_attack_recall"],
                ):
                    best = candidate
    return best


def report_test(y, gate_probability, dl_probabilities, policy, classes):
    benign_index = classes.index("Benign")
    target = y != benign_index
    attack, gated, predicted, confidence = combined_attack(
        gate_probability, dl_probabilities, policy["stage1_threshold"],
        policy["dl_threshold"], benign_index)
    false_positives = int(np.count_nonzero(attack & ~target))
    report = {
        "test_rows": len(y),
        "combined_attack_f1": float(f1_score(target, attack, zero_division=0)),
        "combined_attack_recall": float(recall_score(target, attack, zero_division=0)),
        "combined_false_positive_rate": float(false_positives / np.count_nonzero(~target)),
        "combined_false_positives": false_positives,
        "stage2_reached_count": int(np.count_nonzero(gated)),
        "unknown_after_gate_count": int(np.count_nonzero(
            gated & (confidence < policy["dl_threshold"]))),
        "dl_raw_macro_f1": float(f1_score(y, predicted, average="macro")),
        "by_true_class": {},
    }
    for index, name in enumerate(classes):
        mask = y == index
        if name == "Benign":
            correct_class = (~gated[mask]) | (
                gated[mask] &
                (confidence[mask] >= policy["dl_threshold"]) &
                (predicted[mask] == index))
        else:
            correct_class = (
                gated[mask] &
                (confidence[mask] >= policy["dl_threshold"]) &
                (predicted[mask] == index))
        report["by_true_class"][name] = {
            "support": int(np.count_nonzero(mask)),
            "stage1_forwarded_fraction": float(np.mean(gated[mask])),
            "combined_alert_fraction": float(np.mean(attack[mask])),
            "correct_class_fraction": float(np.mean(correct_class)),
        }
    return report


def main(args):
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    data_path = args.data.resolve()
    dataset_hash = sha256_file(data_path)
    (X, y, encoder, feature_names, ml_features, row_ids,
     train_pos, val_pos, test_pos, row_counts) = prepare_shared_data(
         data_path, args.batch_rows)
    classes = list(encoder.classes_)
    benign_index = classes.index("Benign")
    split_support = {
        name: {label: int(np.count_nonzero(y[positions] == index))
               for index, label in enumerate(classes)}
        for name, positions in (("train", train_pos), ("validation", val_pos),
                                ("test", test_pos))
    }
    print(f"Class support by split: {json.dumps(split_support)}", flush=True)

    stage1_models = fit_stage1(X, y, feature_names, ml_features, train_pos)
    X_train, X_val, X_test, scaler, lower, upper = scale_dl(
        X, train_pos, val_pos, test_pos)
    stage2_model = fit_stage2(
        X_train, X_val, y[train_pos], y[val_pos], classes,
        args.epochs, args.batch_size)
    del X_train
    gc.collect()

    dl_val_probability = stage2_model.predict(
        X_val, batch_size=args.batch_size, verbose=0)
    policy = choose_policy(stage1_models, X, y, feature_names, ml_features,
                           val_pos, dl_val_probability, benign_index)
    print(f"Selected validation policy: {policy}", flush=True)

    # Touch the test set only after selecting model and thresholds on validation.
    selected = stage1_models[policy["model_name"]]
    test_frame = stage1_frames(X, test_pos, feature_names, ml_features)
    stage1_test_probability = selected.predict_proba(test_frame)[:, 1]
    dl_test_probability = stage2_model.predict(
        X_test, batch_size=args.batch_size, verbose=0)
    test_report = report_test(y[test_pos], stage1_test_probability,
                              dl_test_probability, policy, classes)

    ml_dir = output / "ml"
    dl_dir = output / "dl"
    ml_dir.mkdir(parents=True)
    dl_dir.mkdir(parents=True)
    joblib.dump(selected, ml_dir / "stage1_binary_filter.joblib")
    joblib.dump(ml_features, ml_dir / "stage1_feature_list.joblib")
    (ml_dir / "stage1_threshold.json").write_text(
        json.dumps({"attack_threshold": policy["stage1_threshold"]}, indent=2))
    stage2_model.save(dl_dir / "nids_model.keras")
    for filename, artifact in (
        ("scaler.pkl", scaler), ("label_encoder.pkl", encoder),
        ("feature_names.pkl", feature_names),
    ):
        with (dl_dir / filename).open("wb") as destination:
            pickle.dump(artifact, destination)
    metadata = {
        "artifact_version": "shared-split-1",
        "dataset_path": str(data_path),
        "dataset_sha256": dataset_hash,
        "classes": classes,
        "feature_names": feature_names,
        "num_features": len(feature_names),
        "num_classes": len(classes),
        "model_input_shape": [len(feature_names), 1],
        "percentile_clipping": {
            "enabled": True, "lower_percentile": 0.5,
            "upper_percentile": 99.5,
            "lower_bounds": lower.tolist(), "upper_bounds": upper.tolist(),
        },
        "split": {"train_samples": len(train_pos),
                  "validation_samples": len(val_pos),
                  "test_samples": len(test_pos), "random_state": 42},
    }
    (dl_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    (dl_dir / "thresholds.json").write_text(json.dumps({
        "global_threshold": policy["dl_threshold"],
        "metric": "combined_validation_attack_f1",
        "validation_score": policy["validation_attack_f1"],
    }, indent=2))
    np.savez_compressed(
        output / "split_row_ids.npz",
        train=row_ids[train_pos], validation=row_ids[val_pos],
        test=row_ids[test_pos])
    manifest = {
        "dataset_sha256": dataset_hash, "row_counts": row_counts,
        "split_class_support": split_support,
        "policy": policy, "test_report": test_report,
        "versions": {
            "python": platform.python_version(),
            "tensorflow": tf.__version__, "scikit_learn": sklearn.__version__,
            "xgboost": xgb.__version__, "pandas": pd.__version__,
        },
    }
    (output / "shared_evaluation.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(test_report, indent=2), flush=True)
    print(f"Saved shared bundle and report to {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path,
                        default=ROOT / "clean_data" / "cic-collection.parquet")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "updated_models" / "shared")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--batch-rows", type=int, default=50_000,
                        help="Parquet rows read per batch during deduplication")
    main(parser.parse_args())
