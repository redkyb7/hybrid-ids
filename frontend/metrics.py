"""Read a matching combined holdout score for the live dashboard."""

import json
from pathlib import Path


def load_model_metrics(path: Path | None) -> dict[str, float]:
    if path is None or not path.exists():
        return {}
    try:
        with path.open(encoding="utf-8") as metrics_file:
            report = json.load(metrics_file)["test_report"]
        score = float(report["combined_attack_f1"])
        if not 0 <= score <= 1:
            return {}
        return {"combined_attack_f1": score}
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return {}
