"""Measure warmed two-stage inference with the uploaded artifacts.

This forces every synthetic flow through Stage 2. It measures inference only;
packet waiting, capture, SQLite, and detection accuracy are outside this check.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from flow_aggregator import Flow
from hybrid_engine import HybridIDSEngine


def benchmark(runs: int) -> dict:
    engine = HybridIDSEngine(stage1_threshold=0.0)
    features = []
    for index in range(runs):
        flow = Flow("10.0.0.1", "10.0.0.2", 40000 + index, 80, "TCP", 100.0)
        flow.add_packet(0, 100.0, True, {"S": True}, win_size=4096)
        flow.add_packet(20 + index % 20, 100.01, False, {"A": True},
                        win_size=8192, payload_len=20 + index % 20)
        flow.add_packet(40 + index % 40, 100.02, True, {"P": True},
                        payload_len=40 + index % 40)
        features.append(flow.extract_features())

    for item in features[:min(5, runs)]:
        engine.classify_flow(item)

    results = [engine.classify_flow(item) for item in features]
    if any(result["stage_reached"] != "Stage 2 (DL)" for result in results):
        raise AssertionError("A flow did not execute Stage 2")
    latency = np.array([result["latency_ms"] for result in results])
    return {
        "flows": runs,
        "stage2_executed": runs,
        "mean_inference_ms": round(float(np.mean(latency)), 2),
        "p95_inference_ms": round(float(np.percentile(latency, 95)), 2),
        "max_inference_ms": round(float(np.max(latency)), 2),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=100)
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be positive")
    print(json.dumps(benchmark(args.runs), indent=2))
