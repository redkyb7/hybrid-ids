"""Two-stage inference using the selected ML and DL artifact bundle."""

import importlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd


class HybridIDSEngine:
    """Run strict Stage 1 triage followed by the saved Stage 2 predictor."""

    LABELS = {
        "Benign": "Normal Traffic",
        "Botnet": "Botnet",
        "Bruteforce": "Brute Force",
        "DDoS": "DDoS",
        "DoS": "DoS",
        "Infiltration": "Infiltration",
        "Portscan": "Port Scan",
        "Webattack": "Web Attack",
    }

    def __init__(self, artifact_root: str | Path | None = None,
                 stage1_threshold: float | None = None):
        project_root = Path(__file__).resolve().parents[1]
        if artifact_root is None:
            artifact_root = os.environ.get(
                "IDS_ARTIFACT_ROOT",
                project_root / "updated_models" / "extracted",
            )
        artifact_root = Path(artifact_root)
        self.artifact_root = artifact_root

        ml_dir = artifact_root / "ml"
        dl_dir = artifact_root / "dl"

        if stage1_threshold is None:
            configured = os.environ.get("IDS_STAGE1_THRESHOLD")
            saved_threshold = ml_dir / "stage1_threshold.json"
            if configured is not None:
                stage1_threshold = float(configured)
            elif saved_threshold.exists():
                with saved_threshold.open(encoding="utf-8") as source:
                    stage1_threshold = float(json.load(source)["attack_threshold"])
            else:
                stage1_threshold = 0.10
        if not math.isfinite(stage1_threshold) or not 0 <= stage1_threshold <= 1:
            raise ValueError("IDS_STAGE1_THRESHOLD must be between 0 and 1")
        self.stage1_threshold = stage1_threshold

        self.stage1_model = joblib.load(ml_dir / "stage1_binary_filter.joblib")
        self.stage1_features = list(joblib.load(ml_dir / "stage1_feature_list.joblib"))
        if len(self.stage1_features) != len(set(self.stage1_features)):
            raise ValueError("Stage 1 feature list contains duplicates")
        trained_features = getattr(self.stage1_model, "feature_names_in_", None)
        if trained_features is not None and list(trained_features) != self.stage1_features:
            raise ValueError("Stage 1 saved feature order differs from its model")

        classes = list(self.stage1_model.classes_)
        if classes != [0, 1]:
            raise ValueError(f"Expected Stage 1 classes [0, 1], got {classes}")
        self.attack_index = classes.index(1)

        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        predict_module = importlib.import_module(
            "deep learning model.predict"
        )
        self.stage2_classifier = predict_module.NIDSClassifier(
            model_path=str(dl_dir / "nids_model.keras"),
            scaler_path=str(dl_dir / "scaler.pkl"),
            encoder_path=str(dl_dir / "label_encoder.pkl"),
            feature_names_path=str(dl_dir / "feature_names.pkl"),
            metadata_path=str(dl_dir / "metadata.json"),
            thresholds_path=str(dl_dir / "thresholds.json"),
        )
        self.stage2_features = list(self.stage2_classifier.feature_order)
        if set(self.stage2_classifier.label_encoder.classes_) != set(self.LABELS):
            raise ValueError("Stage 2 labels differ from the runtime label map")

    def _validated_features(self, raw_flow_dict: Dict[str, Any]) -> Dict[str, float]:
        required = set(self.stage1_features) | set(self.stage2_features)
        missing = sorted(required - raw_flow_dict.keys())
        if missing:
            raise ValueError(f"Missing model features: {', '.join(missing)}")

        values = {}
        for name in required:
            try:
                value = float(raw_flow_dict[name])
            except (TypeError, ValueError) as error:
                raise ValueError(f"Non-numeric model feature: {name}") from error
            if not math.isfinite(value):
                raise ValueError(f"Non-finite model feature: {name}")
            values[name] = value
        return values

    def classify_flow(self, raw_flow_dict: Dict[str, Any]) -> Dict[str, Any]:
        started = time.perf_counter()
        features = self._validated_features(raw_flow_dict)
        stage1_frame = pd.DataFrame(
            [[features[name] for name in self.stage1_features]],
            columns=self.stage1_features,
        )
        probabilities = np.asarray(self.stage1_model.predict_proba(stage1_frame)[0])
        if probabilities.shape != (2,) or not np.all(np.isfinite(probabilities)):
            raise RuntimeError("Stage 1 returned invalid probabilities")
        attack_probability = float(probabilities[self.attack_index])
        stage1_ms = (time.perf_counter() - started) * 1000

        if attack_probability < self.stage1_threshold:
            return {
                "verdict": "BENIGN",
                "attack_type": "Normal Traffic",
                "confidence": round(1 - attack_probability, 4),
                "stage_reached": "Stage 1 (ML)",
                "stage1_attack_probability": attack_probability,
                "stage1_latency_ms": round(stage1_ms, 2),
                "stage2_latency_ms": 0.0,
                "latency_ms": max(1, round((time.perf_counter() - started) * 1000, 2)),
            }

        stage2_started = time.perf_counter()
        prediction = self.stage2_classifier.predict_detailed(features)
        label = prediction["label"]
        if label == "Unknown":
            # Stage 1 detected an attack candidate. Preserve the alert while
            # leaving the class unassigned instead of calling it Benign.
            verdict = "MALICIOUS"
            attack_type = "Unknown Attack"
            confidence = attack_probability
        elif label == "Benign":
            verdict = "BENIGN"
            attack_type = self.LABELS[label]
            confidence = prediction["confidence"]
        else:
            verdict = "MALICIOUS"
            attack_type = self.LABELS[label]
            confidence = prediction["confidence"]

        return {
            "verdict": verdict,
            "attack_type": attack_type,
            "confidence": round(float(confidence), 4),
            "stage_reached": "Stage 2 (DL)",
            "model_class": label,
            "stage1_attack_probability": attack_probability,
            "stage1_latency_ms": round(stage1_ms, 2),
            "stage2_latency_ms": round((time.perf_counter() - stage2_started) * 1000, 2),
            "latency_ms": max(1, round((time.perf_counter() - started) * 1000, 2)),
        }
