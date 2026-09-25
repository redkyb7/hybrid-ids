"""Compare one labeled lab campaign with the monitor's recorded verdicts.

Example: uv tool run --from duckdb python scripts/evaluate_attack_campaigns.py
         --campaign-id 20260925T120000Z-1234abcd
This script uses only Python's standard library; the uv command supplies a
Python interpreter on hosts where ``python`` is not on PATH.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3


ROOT = Path(__file__).resolve().parents[1]
DISPLAY_LABELS = {
    "Botnet": "Botnet",
    "Bruteforce": "Brute Force",
    "DDoS": "DDoS",
    "DoS": "DoS",
    "Infiltration": "Infiltration",
    "Portscan": "Port Scan",
    "Webattack": "Web Attack",
}


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return round(
        ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower), 4
    )


def utc_sql(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def compare_features(vectors: list[dict], reference: dict) -> dict:
    comparisons = {}
    if not vectors:
        return comparisons
    for feature, dataset_stat in reference.get("features", {}).items():
        values = [float(vector[feature]) for vector in vectors if feature in vector]
        if not values:
            continue
        live_median = percentile(values, 0.5)
        low_bound, high_bound = dataset_stat["p10"], dataset_stat["p90"]
        comparisons[feature] = {
            "live_median": live_median,
            "dataset_p10": low_bound,
            "dataset_median": dataset_stat["median"],
            "dataset_p90": high_bound,
            "median_within_dataset_p10_p90": (
                low_bound is not None and high_bound is not None
                and low_bound <= live_median <= high_bound
            ),
        }
    return comparisons


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--db", type=Path, default=ROOT / "data/ids_logs.db")
    parser.add_argument("--manifests", type=Path, default=ROOT / "data/campaigns")
    parser.add_argument("--profile", type=Path, default=ROOT / "analysis/attack_dataset_profile.json")
    args = parser.parse_args()
    paths = sorted(
        path for path in args.manifests.glob(f"{args.campaign_id}-*.json")
        if path.name != f"{args.campaign_id}-evaluation.json"
    )
    if not paths:
        parser.error(f"no manifests found for {args.campaign_id}")
    if not args.db.exists():
        parser.error(f"monitor database not found: {args.db}")
    manifests = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    modes = {item["mode"] for item in manifests}
    classes = {item["class_label"] for item in manifests}
    sources = {item["source_ip"] for item in manifests}
    if len(modes) != 1 or len(classes) != 1:
        parser.error("workers in one campaign have different modes or broad labels")
    class_label = next(iter(classes))
    mode = next(iter(modes))
    start_epoch = min(item["started_epoch"] for item in manifests)
    end_epoch = max(item["ended_epoch"] for item in manifests)
    low = utc_sql(start_epoch - 1)
    high = utc_sql(end_epoch + 3)
    placeholders = ",".join("?" for _ in sources)
    with sqlite3.connect(args.db) as connection:
        connection.execute("PRAGMA query_only=ON")
        columns = {row[1] for row in connection.execute("PRAGMA table_info(logs)")}
        optional = [
            column if column in columns else f"NULL AS {column}"
            for column in ("flow_features_json", "flow_start_epoch", "flow_end_epoch")
        ]
        window_clause = (
            "(flow_end_epoch BETWEEN ? AND ? OR "
            "(flow_end_epoch IS NULL AND timestamp BETWEEN ? AND ?))"
            if "flow_end_epoch" in columns else "timestamp BETWEEN ? AND ?"
        )
        query = (
            "SELECT id, source_ip, destination_ip, protocol, source_port, "
            "destination_port, timestamp, stage_reached, "
            "model_attack_type, model_verdict, "
            "stage1_attack_probability, stage1_features_json, latency_ms, "
            + ", ".join(optional) + " FROM logs WHERE source_ip IN ("
            + placeholders + ") AND destination_ip = ? AND "
            + window_clause + " ORDER BY id"
        )
        window_args = (
            [start_epoch - 1, end_epoch + 3, low, high]
            if "flow_end_epoch" in columns else [low, high]
        )
        cursor = connection.execute(
            query, [*sources, "192.168.100.10", *window_args]
        )
        names = [description[0] for description in cursor.description]
        rows = [dict(zip(names, raw)) for raw in cursor.fetchall()]

    observed_sources = {row["source_ip"] for row in rows}
    def connection_key(row):
        return (row["source_ip"], row["destination_ip"], row["protocol"],
                row["source_port"], row["destination_port"])

    def captured_packets(item):
        encoded = item["flow_features_json"] or item["stage1_features_json"]
        if not encoded:
            return 0
        features = json.loads(encoded)
        return (features.get("Total Fwd Packets", 0)
                + features.get("Total Backward Packets", 0))

    connections = {}
    for row in rows:
        key = connection_key(row)
        # A late TCP teardown packet can appear after the completed exchange
        # as a fresh one-packet flow with the same 5-tuple. Compare the
        # fullest captured snapshot rather than that misleading last record.
        previous = connections.get(key)
        if previous is None or captured_packets(row) >= captured_packets(previous):
            connections[key] = row
    stage2 = [row for row in rows if row["stage_reached"] == "Stage 2 (DL)"]
    model_alerts = [row for row in rows if row["model_verdict"] == "MALICIOUS"]
    correct_class = [
        row for row in rows
        if row["model_verdict"] == "MALICIOUS"
        and row["model_attack_type"] == DISPLAY_LABELS[class_label]
    ]
    scores = [float(row["stage1_attack_probability"]) for row in rows
              if row["stage1_attack_probability"] is not None]
    latency = [float(row["latency_ms"]) for row in rows
               if row["latency_ms"] is not None]
    def feature_vector(row):
        encoded = row["flow_features_json"] or row["stage1_features_json"]
        return json.loads(encoded) if encoded else None

    feature_vectors = [vector for row in rows
                       if (vector := feature_vector(row)) is not None]
    fullest_vectors = [vector for row in connections.values()
                       if (vector := feature_vector(row)) is not None]
    reference = {}
    if args.profile.exists():
        profiles = json.loads(args.profile.read_text(encoding="utf-8"))
        reference_label = manifests[0].get("reference_label")
        reference = profiles["subtypes"].get(reference_label, {}) if reference_label else {}
    comparisons = compare_features(fullest_vectors, reference)
    snapshot_comparisons = compare_features(feature_vectors, reference)

    invalid_reasons = []
    if any(item["status"] != "completed" for item in manifests):
        invalid_reasons.append("one or more worker manifests are incomplete")
    if "unknown" in sources:
        invalid_reasons.append("a worker source IP is unknown")
    if not rows:
        invalid_reasons.append("monitor recorded no matching snapshots")
    if class_label == "DDoS" and len(observed_sources) < 2:
        invalid_reasons.append("fewer than two distinct worker source IPs were observed")
    result = {
        "campaign_id": args.campaign_id,
        "mode": mode,
        "class_label": class_label,
        "reference_label": manifests[0].get("reference_label"),
        "ground_truth_valid": not invalid_reasons,
        "invalid_reasons": invalid_reasons,
        "manifest_files": [path.name for path in paths],
        "worker_source_ips": sorted(sources),
        "observed_source_ips": sorted(observed_sources),
        "window_utc": {"start": low, "end": high},
        "actions_attempted": sum(item["actions_attempted"] for item in manifests),
        "actions_succeeded": sum(item["actions_succeeded"] for item in manifests),
        "snapshots": len(rows),
        "distinct_connections": len(connections),
        "stage2_snapshots": len(stage2),
        "stage2_connections": len({connection_key(row) for row in stage2}),
        "model_alert_snapshots": len(model_alerts),
        "model_alert_connections": len({connection_key(row) for row in model_alerts}),
        "model_correct_class_snapshots": len(correct_class),
        "model_correct_class_connections": len({connection_key(row) for row in correct_class}),
        "stage1_attack_score_median": percentile(scores, 0.5),
        "stage1_attack_score_p90": percentile(scores, 0.9),
        "latency_ms_p95": percentile(latency, 0.95),
        "stage1_feature_vectors_logged": len(feature_vectors),
        "fullest_snapshot_vectors_compared": len(fullest_vectors),
        "fullest_snapshot_feature_comparisons": comparisons,
        "all_snapshot_feature_comparisons": snapshot_comparisons,
    }
    output = args.manifests / f"{args.campaign_id}-evaluation.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"{mode} | {class_label} | valid={result['ground_truth_valid']}")
    print(f"workers observed: {len(observed_sources)} | actions: "
          f"{result['actions_succeeded']}/{result['actions_attempted']} | "
          f"connections: {len(connections)} | snapshots: {len(rows)}")
    print(f"Stage 2: {len(stage2)} snapshots/{result['stage2_connections']} connections | "
          f"model alerts: {len(model_alerts)} | correct model class: {len(correct_class)} | "
          f"p95 latency: {result['latency_ms_p95']} ms")
    if comparisons:
        matched = sum(value["median_within_dataset_p10_p90"] for value in comparisons.values())
        print(f"Fullest-snapshot feature medians within CIC p10-p90: {matched}/{len(comparisons)}")
    if invalid_reasons:
        print("Evaluation limitations: " + "; ".join(invalid_reasons))
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
