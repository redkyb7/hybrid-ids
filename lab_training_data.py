"""Read verified, campaign-separated lab training partitions.

The Colab bundle contains only train and validation Parquets. The final
holdout stays outside the bundle until a candidate pipeline is frozen.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


LAB_CLASSES = {"Benign", "Botnet", "Bruteforce", "DDoS", "DoS", "Portscan"}


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_lab_split(plan_path: str | Path, split: str) -> tuple[pd.DataFrame, list[str]]:
    if split not in {"train", "validation"}:
        raise ValueError("Only train and validation lab splits are available to training")
    plan_path = Path(plan_path).resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("schema_version") != 1 or set(plan.get("splits", {})) != {
        "train", "validation"
    }:
        raise ValueError("Expected a training-only lab split plan")
    features = plan.get("features")
    if not isinstance(features, list) or len(features) != 57 or len(set(features)) != 57:
        raise ValueError("Expected exactly 57 distinct model features")
    if hashlib.sha256(json.dumps(features).encode()).hexdigest() != plan.get(
        "feature_schema_sha256"
    ):
        raise ValueError("Lab feature schema hash differs from the split plan")
    if set(plan["splits"]["train"]["campaign_ids"]) & set(
        plan["splits"]["validation"]["campaign_ids"]
    ):
        raise ValueError("Lab train and validation campaigns overlap")
    stats = plan["splits"][split]
    source = (plan_path.parent / stats["parquet"]).resolve()
    if hashlib.sha256(source.read_bytes()).hexdigest() != stats["parquet_sha256"]:
        raise ValueError(f"Lab {split} Parquet hash differs from the split plan")
    extra = ["connection_id"] if "connection_ids" in stats else []
    frame = pd.read_parquet(source, columns=["campaign_id", *extra, "ClassLabel", *features])
    if extra:
        if set(frame['connection_id']) != set(stats['connection_ids']):
            raise ValueError('Connection IDs differ from split plan')
        if set(plan['splits']['train']['connection_ids']) & set(plan['splits']['validation']['connection_ids']):
            raise ValueError('TCP/UDP connections overlap between live partitions')
    if len(frame) != stats["usable_rows"]:
        raise ValueError(f"Lab {split} row count differs from the split plan")
    if set(frame["campaign_id"]) != set(stats["campaign_ids"]):
        raise ValueError(f"Lab {split} campaign IDs differ from the split plan")
    if set(frame["ClassLabel"]) != LAB_CLASSES:
        raise ValueError(f"Lab {split} must contain all six classes")
    if not np.isfinite(frame[features].to_numpy(dtype=np.float64)).all():
        raise ValueError(f"Lab {split} contains non-finite model feature values")
    return frame[["ClassLabel", *features]], features
