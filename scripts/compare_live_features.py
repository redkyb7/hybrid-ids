"""Compare logged Stage 1 live inputs with a stratified CIC Parquet sample.

Requires pyarrow in the analysis environment. This reads the dataset in batches
and never loads its nine million rows into memory at once.
"""

import argparse
import json
import sqlite3
from pathlib import Path

import joblib
import numpy as np
import pyarrow.parquet as parquet


TRAINING_CLASS = {
    "benign": "Benign",
    "scan": "Portscan",
    "dos": "DoS",
    "bruteforce": "Bruteforce",
    "web": "Webattack",
    "botnet": "Botnet",
}
SAMPLE_STRIDE = {
    "Benign": 1000,
    "Portscan": 1,
    "DoS": 250,
    "Bruteforce": 50,
    "Webattack": 1,
    "Botnet": 50,
}


def describe(values):
    return {
        "p10": float(np.percentile(values, 10)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "zero_fraction": float(np.mean(values == 0)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    feature_names = list(joblib.load(args.features))
    windows = json.loads(args.windows.read_text(encoding="utf-8"))
    live = {}
    with sqlite3.connect(args.db) as connection:
        for name, bounds in windows.items():
            rows = connection.execute(
                "SELECT stage1_features_json FROM logs WHERE id > ? AND id <= ? "
                "AND stage1_features_json IS NOT NULL",
                (bounds[0], bounds[1]),
            ).fetchall()
            if not rows:
                raise ValueError(f"No logged feature vectors for {name}")
            live[name] = np.asarray([
                [json.loads(row[0])[feature] for feature in feature_names]
                for row in rows
            ], dtype=np.float64)

    selected = {name: [] for name in SAMPLE_STRIDE}
    source = parquet.ParquetFile(args.parquet)
    offset = 0
    for batch in source.iter_batches(batch_size=50_000,
                                     columns=feature_names + ["ClassLabel"]):
        frame = batch.to_pandas()
        labels = frame["ClassLabel"].to_numpy()
        for label, stride in SAMPLE_STRIDE.items():
            positions = np.flatnonzero(labels == label)
            positions = positions[(positions + offset) % stride == 0]
            if len(positions):
                selected[label].append(frame.iloc[positions][feature_names].to_numpy(dtype=np.float64))
        offset += len(frame)
        if offset % 1_000_000 < len(frame):
            print(f"Scanned {offset:,} Parquet rows", flush=True)

    training = {name: np.concatenate(parts) for name, parts in selected.items()}
    report = {
        "parquet_rows_scanned": offset,
        "sampling": "Deterministic row-position stride within each broad class",
        "training_sample_counts": {name: len(values) for name, values in training.items()},
        "live_counts": {name: len(values) for name, values in live.items()},
        "comparisons": {},
    }
    for name, values in live.items():
        reference = training[TRAINING_CLASS[name]]
        rows = []
        for index, feature in enumerate(feature_names):
            train_values = reference[:, index]
            live_values = values[:, index]
            train_stats = describe(train_values)
            rows.append({
                "feature": feature,
                "training": train_stats,
                "live": describe(live_values),
                "live_outside_training_p10_p90_fraction": float(np.mean(
                    (live_values < train_stats["p10"]) |
                    (live_values > train_stats["p90"])
                )),
            })
        report["comparisons"][name] = rows
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "parquet_rows_scanned": report["parquet_rows_scanned"],
        "training_sample_counts": report["training_sample_counts"],
        "live_counts": report["live_counts"],
        "output": str(args.output),
    }))


if __name__ == "__main__":
    main()
