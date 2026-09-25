"""Profile the CIC training Parquet and extract bounded, reviewable attack rows.

Requires DuckDB (for example: ``uv tool run --from duckdb python scripts/profile_attack_dataset.py``).
The full attack-row export and CSV samples are local analysis data, ignored by
Git. The compact JSON profile can be reviewed and versioned with the plan.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import duckdb


ROOT = Path(__file__).resolve().parents[1]
FEATURES = [
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Fwd Packets Length Total",
    "Bwd Packets Length Total",
    "Fwd Packet Length Max",
    "Fwd Packet Length Mean",
    "Bwd Packet Length Max",
    "Bwd Packet Length Mean",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Fwd PSH Flags",
    "Fwd Header Length",
    "Bwd Header Length",
    "Packet Length Max",
    "Packet Length Mean",
    "Packet Length Variance",
    "SYN Flag Count",
    "URG Flag Count",
    "Init Fwd Win Bytes",
    "Init Bwd Win Bytes",
    "Fwd Seg Size Min",
    "Active Std",
    "Idle Std",
]
EXAMPLE_FEATURES = [
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Fwd Packets Length Total",
    "Bwd Packets Length Total",
    "Packet Length Max",
]


def sql_string(value: object) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def csv_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", value) + ".csv"


def auc(positive: list[dict], negative: list[dict], feature: str) -> float | None:
    """Mann-Whitney AUC with average ranks for ties; missing values omitted."""
    values = []
    for rows, label in ((positive, 1), (negative, 0)):
        for row in rows:
            try:
                number = float(row[feature])
            except (ValueError, TypeError):
                continue
            if math.isfinite(number):
                values.append((number, label))
    values.sort(key=lambda item: item[0])
    positives = sum(label for _, label in values)
    negatives = len(values) - positives
    if not positives or not negatives:
        return None
    positive_rank_sum = 0.0
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[end][0] == values[start][0]:
            end += 1
        average_rank = (start + 1 + end) / 2
        positive_rank_sum += average_rank * sum(label for _, label in values[start:end])
        start = end
    return round(
        (positive_rank_sum - positives * (positives + 1) / 2)
        / (positives * negatives),
        4,
    )


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def representative_row(rows: list[dict], profile: dict) -> dict:
    """Choose the sampled row nearest the subtype's full-data medians."""
    def signed_log(number: float) -> float:
        return math.copysign(math.log1p(abs(number)), number)

    def distance(row: dict) -> float:
        score = 0.0
        for feature in EXAMPLE_FEATURES:
            median = profile["features"][feature]["median"]
            number = float(row[feature])
            if median is None or not math.isfinite(number):
                continue
            score += abs(signed_log(number) - signed_log(float(median)))
        return score

    chosen = min(rows, key=distance)
    result = {}
    for column, value in chosen.items():
        if column in ("ClassLabel", "Label"):
            result[column] = value
        else:
            try:
                number = float(value)
                result[column] = number if math.isfinite(number) else None
            except (ValueError, TypeError):
                result[column] = None
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parquet", type=Path, default=ROOT / "clean_data/cic-collection.parquet"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "analysis")
    parser.add_argument("--broad-sample", type=int, default=2000)
    parser.add_argument("--subtype-sample", type=int, default=100)
    parser.add_argument(
        "--reuse-exports", action="store_true",
        help="Reuse previously exported attack Parquet files and refresh profiles/samples",
    )
    args = parser.parse_args()
    source = args.parquet.resolve(strict=True)
    output = args.output.resolve()
    rows_dir = output / "attack_rows"
    samples_dir = output / "attack_row_samples"
    if args.broad_sample < 1 or args.subtype_sample < 1:
        parser.error("sample sizes must be positive")
    if rows_dir.exists() and any(rows_dir.iterdir()) and not args.reuse_exports:
        parser.error(f"{rows_dir} already contains data; choose a new --output")
    if samples_dir.exists() and any(samples_dir.iterdir()) and not args.reuse_exports:
        parser.error(f"{samples_dir} already contains data; choose a new --output")
    if args.reuse_exports and not list(rows_dir.glob("ClassLabel=*/Label=*/*.parquet")):
        parser.error(f"{rows_dir} has no attack-row export to reuse")
    output.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)

    connection = duckdb.connect()
    # A fixed seed is repeatable with a single query worker.
    connection.execute("SET threads = 1")
    source_sql = sql_string(source.as_posix())
    columns = [row[0] for row in connection.execute(
        f"DESCRIBE SELECT * FROM read_parquet({source_sql})"
    ).fetchall()]
    missing = set(FEATURES + ["Label", "ClassLabel"]) - set(columns)
    if missing:
        parser.error(f"missing expected dataset columns: {sorted(missing)}")

    expressions = ["ClassLabel", "Label", "GROUPING(Label) AS broad_group", "count(*) AS rows"]
    for index, feature in enumerate(FEATURES):
        finite = f'CASE WHEN isfinite(CAST("{feature}" AS DOUBLE)) THEN CAST("{feature}" AS DOUBLE) END'
        expressions.extend([
            f"approx_quantile({finite}, [0.1, 0.5, 0.9]) AS q{index}",
            f'avg(CASE WHEN "{feature}" = 0 THEN 1.0 ELSE 0.0 END) AS z{index}',
            f'avg(CASE WHEN NOT isfinite(CAST("{feature}" AS DOUBLE)) THEN 1.0 ELSE 0.0 END) AS nf{index}',
        ])
    query = (
        f"SELECT {', '.join(expressions)} FROM read_parquet({source_sql}) "
        "GROUP BY GROUPING SETS ((ClassLabel), (ClassLabel, Label)) "
        "ORDER BY ClassLabel, broad_group DESC, Label"
    )
    print("Computing full-dataset class and subtype profiles...", flush=True)
    cursor = connection.execute(query)
    names = [item[0] for item in cursor.description]
    broad, subtypes = {}, {}
    for raw in cursor.fetchall():
        record = dict(zip(names, raw))
        entry = {"rows": record["rows"], "features": {}}
        for index, feature in enumerate(FEATURES):
            quantiles = record[f"q{index}"]
            entry["features"][feature] = {
                "p10": quantiles[0] if quantiles else None,
                "median": quantiles[1] if quantiles else None,
                "p90": quantiles[2] if quantiles else None,
                "zero_fraction": record[f"z{index}"],
                "nonfinite_fraction": record[f"nf{index}"],
            }
        if record["broad_group"]:
            broad[record["ClassLabel"]] = entry
        else:
            entry["class"] = record["ClassLabel"]
            subtypes[record["Label"]] = entry
    attack_total = sum(group["rows"] for label, group in broad.items() if label != "Benign")
    if not args.reuse_exports:
        print(f"Exporting {attack_total:,} attack rows by broad class and subtype...", flush=True)
        connection.execute(
            f"COPY (SELECT * FROM read_parquet({source_sql}) "
            "WHERE ClassLabel <> 'Benign') "
            f"TO {sql_string(rows_dir.as_posix())} "
            "(FORMAT PARQUET, PARTITION_BY (ClassLabel, Label))"
        )
    exported_pattern = rows_dir / "ClassLabel=*" / "Label=*" / "*.parquet"
    exported_counts = {
        (class_name, label): count
        for class_name, label, count in connection.execute(
            "SELECT ClassLabel, Label, count(*) "
            f"FROM read_parquet({sql_string(exported_pattern.as_posix())}, "
            "hive_partitioning=true) GROUP BY 1, 2"
        ).fetchall()
    }
    expected_counts = {
        (entry["class"], label): entry["rows"]
        for label, entry in subtypes.items()
        if entry["class"] != "Benign"
    }
    if exported_counts != expected_counts:
        raise RuntimeError("exported attack-row counts differ from source subtype counts")
    print(f"Verified {sum(exported_counts.values()):,} rows across {len(exported_counts)} subtypes", flush=True)

    sample_rows = {}
    for label, group in broad.items():
        amount = min(args.broad_sample, group["rows"])
        if label == "Benign":
            relation = (
                f"(SELECT * FROM read_parquet({source_sql}) "
                "WHERE ClassLabel = 'Benign')"
            )
        else:
            pattern = rows_dir / f"ClassLabel={label}" / "Label=*" / "*.parquet"
            relation = f"read_parquet({sql_string(pattern.as_posix())}, hive_partitioning=true)"
        target = samples_dir / f"broad_{csv_name(label)}"
        if target.exists():
            target.unlink()
        connection.execute(
            f"COPY (SELECT * FROM {relation} USING SAMPLE "
            f"reservoir({amount} ROWS) REPEATABLE (42)) "
            f"TO {sql_string(target.as_posix())} (FORMAT CSV, HEADER TRUE)"
        )
        sample_rows[label] = read_csv(target)
        if len(sample_rows[label]) != amount:
            raise RuntimeError(f"expected {amount} sampled rows for {label}")
        print(f"Sampled {label}: {len(sample_rows[label]):,} rows", flush=True)

    examples = {}
    for label, group in subtypes.items():
        if group["class"] == "Benign":
            continue
        amount = min(args.subtype_sample, group["rows"])
        pattern = (
            rows_dir / f"ClassLabel={group['class']}" / f"Label={label}" / "*.parquet"
        )
        target = samples_dir / f"subtype_{csv_name(label)}"
        if target.exists():
            target.unlink()
        connection.execute(
            f"COPY (SELECT * FROM read_parquet({sql_string(pattern.as_posix())}, "
            f"hive_partitioning=true) USING SAMPLE reservoir({amount} ROWS) "
            f"REPEATABLE (42)) TO {sql_string(target.as_posix())} "
            "(FORMAT CSV, HEADER TRUE)"
        )
        subtype_rows = read_csv(target)
        if len(subtype_rows) != amount:
            raise RuntimeError(f"expected {amount} sampled rows for {label}")
        examples[label] = {
            "class": group["class"],
            "sampled_rows": amount,
            "row": representative_row(subtype_rows, group),
        }

    attack_labels = [label for label in broad if label != "Benign"]
    for label in attack_labels:
        current = sample_rows[label]
        other_attacks = [
            row for other in attack_labels if other != label for row in sample_rows[other]
        ]
        broad[label]["sample_auc_vs_benign"] = {
            feature: auc(current, sample_rows["Benign"], feature) for feature in FEATURES
        }
        broad[label]["sample_auc_vs_other_attacks"] = {
            feature: auc(current, other_attacks, feature) for feature in FEATURES
        }

    report = {
        "dataset": source.name,
        "dataset_bytes": source.stat().st_size,
        "rows": sum(group["rows"] for group in broad.values()),
        "attack_rows": attack_total,
        "method": {
            "quantiles": "DuckDB approx_quantile on all rows, finite values only",
            "zero_fraction": "exact fraction of zero values on all rows",
            "auc": "univariate rank AUC from seeded reservoir samples; 0.5 means no separation",
            "sample_threads": 1,
            "sample_seed": 42,
            "broad_sample_per_class": args.broad_sample,
            "subtype_sample_per_label": args.subtype_sample,
            "attack_export": str(rows_dir.relative_to(output)),
            "sample_export": str(samples_dir.relative_to(output)),
            "exported_subtypes_verified": len(exported_counts),
        },
        "broad_classes": broad,
        "subtypes": subtypes,
    }
    destination = output / "attack_dataset_profile.json"
    destination.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Wrote {destination}", flush=True)
    examples_path = output / "attack_row_examples.json"
    examples_path.write_text(
        json.dumps({
            "selection": "Nearest full-dataset subtype medians over six core flow features within a seeded 100-row reservoir sample (or all rows if fewer)",
            "subtypes": examples,
        }, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {examples_path}", flush=True)


if __name__ == "__main__":
    main()
