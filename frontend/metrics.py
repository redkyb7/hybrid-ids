"""Read the selected DL model's standalone classification score."""

import json
from pathlib import Path


def load_dl_metrics(path: Path | None) -> dict[str, float]:
    if path is None or not path.exists():
        return {}
    try:
        with path.open(encoding="utf-8") as metrics_file:
            report = json.load(metrics_file)
        score = float(report["macro_f1"])
        if not 0 <= score <= 1:
            return {}
        return {"dl_macro_f1": score}
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return {}
