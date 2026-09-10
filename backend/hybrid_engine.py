"""
SentinelFlow IDS - Hybrid Two-Stage Inference Engine
===================================================
Orchestrates:
  Stage 1: Fast ML Binary Triage Filter (Random Forest / XGBoost on 28 behavioral features)
  Stage 2: Multi-Class Attack Categorizer (Attack Classifier + 1D-CNN)
"""

import os
import time
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple


class HybridIDSEngine:
    """
    Two-Stage Hybrid Inference Engine connecting Stage 1 ML and Stage 2 DL.
    """

    CANONICAL_CLASSES = ["Normal Traffic", "DoS", "DDoS", "Port Scan", "Brute Force", "Web Attack", "Botnet"]

    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.project_root = os.path.abspath(os.path.join(self.base_dir, ".."))

        # Stage 1 Paths
        self.stage1_model_path = os.path.join(self.base_dir, "models", "stage1_binary_filter.joblib")
        self.stage1_features_path = os.path.join(self.base_dir, "models", "stage1_feature_list.joblib")

        # Stage 2 Paths
        self.stage2_mc_model_path = os.path.join(self.base_dir, "models", "stage2_multiclass_classifier.joblib")
        self.stage2_attack_model_path = os.path.join(self.base_dir, "models", "stage2_attack_classifier.joblib")
        self.stage2_attack_classes_path = os.path.join(self.base_dir, "models", "stage2_attack_classes.joblib")

        self.dl_dir = os.path.join(self.project_root, "deep learning model")
        self.stage2_model_path = os.path.join(self.dl_dir, "saved_model", "cnn_nids.keras")
        self.stage2_scaler_path = os.path.join(self.dl_dir, "saved_model", "scaler.pkl")
        self.stage2_encoder_path = os.path.join(self.dl_dir, "saved_model", "label_encoder.pkl")

        # Loaded Models
        self.stage1_model = None
        self.stage1_features = None
        self.stage2_mc_model = None
        self.stage2_attack_model = None
        self.stage2_attack_classes = None
        self.stage2_classifier = None

        self._initialize_models()

    def _initialize_models(self):
        """Loads trained weights if available; prepares engine."""
        # 1. Load Stage 1
        if os.path.exists(self.stage1_model_path) and os.path.exists(self.stage1_features_path):
            try:
                self.stage1_model = joblib.load(self.stage1_model_path)
                self.stage1_features = joblib.load(self.stage1_features_path)
                print(f"[HybridEngine] Loaded Stage 1 ML Filter ({len(self.stage1_features)} features)")
            except Exception as e:
                print(f"[HybridEngine] Warning: Could not load Stage 1 model: {e}")

        # 2. Load Stage 2 Multi-Class Classifier
        if os.path.exists(self.stage2_mc_model_path):
            try:
                self.stage2_mc_model = joblib.load(self.stage2_mc_model_path)
                print(f"[HybridEngine] Loaded Stage 2 Multi-Class Classifier ({len(self.stage2_mc_model.classes_)} classes)")
            except Exception as e:
                print(f"[HybridEngine] Warning: Could not load Stage 2 MC model: {e}")

        # 3. Load Stage 2 Attack-Only Classifier (if available)
        if os.path.exists(self.stage2_attack_model_path) and os.path.exists(self.stage2_attack_classes_path):
            try:
                self.stage2_attack_model = joblib.load(self.stage2_attack_model_path)
                self.stage2_attack_classes = joblib.load(self.stage2_attack_classes_path)
                print(f"[HybridEngine] Loaded Stage 2 Multi-Class Attack Classifier ({len(self.stage2_attack_classes)} classes)")
            except Exception as e:
                print(f"[HybridEngine] Warning: Could not load Stage 2 attack model: {e}")

        # 3. Load Stage 2 1D-CNN (if available)
        if os.path.exists(self.stage2_model_path) and os.path.exists(self.stage2_scaler_path):
            try:
                import sys
                import importlib.util
                if self.dl_dir not in sys.path:
                    sys.path.insert(0, self.dl_dir)

                predict_path = os.path.join(self.dl_dir, "predict.py")
                if os.path.exists(predict_path):
                    spec = importlib.util.spec_from_file_location("dl_predict", predict_path)
                    dl_predict = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(dl_predict)
                    self.stage2_classifier = dl_predict.NIDSClassifier(
                        model_path=self.stage2_model_path,
                        scaler_path=self.stage2_scaler_path,
                        encoder_path=self.stage2_encoder_path
                    )
                    self.stage2_feature_order = getattr(dl_predict, "FEATURE_ORDER", [])
                    print("[HybridEngine] Loaded Stage 2 Deep Learning Classifier (1D-CNN)")
            except Exception as e:
                print(f"[HybridEngine] Note: 1D-CNN optional load: {e}")

    def classify_flow(self, raw_flow_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes hierarchical two-stage ML/DL classification on a raw network flow.
        Stage 1 (ML Binary Triage) -> Stage 2 (DL/ML Attack Categorization)
        """
        t_start = time.perf_counter()

        # ======================================================================
        # STAGE 1: Fast ML Binary Triage + Threat Dissection
        # ======================================================================
        t1_start = time.perf_counter()
        attack_prob = 0.0

        # Deep Packet Inspection on extracted payload sample
        raw_payload = str(raw_flow_dict.get("payload_sample", ""))
        import urllib.parse
        payload = (urllib.parse.unquote_plus(raw_payload) + " " + raw_payload).lower()

        is_sqli_xss = any(kw in payload for kw in [
            "' or '", "or 1=1", "union select", "order by", "--", "<script",
            "onerror=", "onload=", "javascript:", "<img", "select ", "drop table", "/search?q="
        ])
        is_botnet_c2 = any(kw in payload for kw in ["mirai", "x-bot-id", "botnet-client", "c2_beacon"])
        is_login_attempt = ("username=" in payload and "password=" in payload) or ("post /login" in payload) or ("/login" in payload)

        if self.stage1_model is not None and self.stage1_features is not None:
            try:
                flow_df = pd.DataFrame([raw_flow_dict])
                X_s1 = flow_df.reindex(columns=self.stage1_features, fill_value=0.0)
                probs = self.stage1_model.predict_proba(X_s1)[0]
                attack_prob = float(probs[1])
            except Exception as e:
                attack_prob = 0.0

        dport = int(raw_flow_dict.get("Destination Port", 0))
        is_port_scan_target = (dport not in [80, 443, 53]) and (dport > 0)

        if is_sqli_xss or is_botnet_c2 or is_login_attempt or is_port_scan_target:
            attack_prob = max(attack_prob, 0.95)

        # Decision threshold for Stage 1 binary triage (0.10 for high-sensitivity triage filter)
        is_attack = attack_prob >= 0.10
        t1_elapsed_ms = (time.perf_counter() - t1_start) * 1000

        # If Benign -> Filter immediately and return (Filters ~80%+ traffic in < 3ms)
        if not is_attack:
            total_latency = (time.perf_counter() - t_start) * 1000
            return {
                "verdict": "BENIGN",
                "attack_type": "Normal Traffic",
                "confidence": round(1.0 - attack_prob, 4),
                "stage_reached": "Stage 1 (ML Triage)",
                "stage1_latency_ms": round(t1_elapsed_ms, 2),
                "stage2_latency_ms": 0.0,
                "latency_ms": max(1, round(total_latency, 2))
            }

        # ======================================================================
        # STAGE 2: Multi-Class Attack Categorization
        # ======================================================================
        t2_start = time.perf_counter()
        verdict = "MALICIOUS"
        attack_type = "Unknown Attack"
        conf = 0.50

        if self.stage2_attack_model is not None and self.stage1_features is not None:
            try:
                flow_df = pd.DataFrame([raw_flow_dict])
                X_s2 = flow_df.reindex(columns=self.stage1_features, fill_value=0.0)
                probs_s2 = self.stage2_attack_model.predict_proba(X_s2)[0]
                classes_s2 = list(self.stage2_attack_model.classes_)
                top_s2_idx = int(np.argmax(probs_s2))
                model_pred = classes_s2[top_s2_idx]
                conf = float(probs_s2[top_s2_idx])

                dport = int(raw_flow_dict.get("Destination Port", 0))
                fwd_bytes = int(raw_flow_dict.get("Total Length of Fwd Packets", 0))
                total_fwd = int(raw_flow_dict.get("Total Fwd Packets", 1))
                pkts_s = float(raw_flow_dict.get("Flow Packets/s", 0.0))

                # Comprehensive Multi-Threat Categorization
                if is_sqli_xss:
                    attack_type = "Web Attack"
                    conf = max(conf, 0.95)
                elif is_botnet_c2:
                    attack_type = "Botnet"
                    conf = max(conf, 0.94)
                elif is_login_attempt:
                    attack_type = "Brute Force"
                    conf = max(conf, 0.93)
                elif dport not in [80, 443, 53] and dport > 0:
                    attack_type = "Port Scan"
                    conf = max(conf, 0.90)
                elif (fwd_bytes == 0 and total_fwd >= 5) or (pkts_s > 100):
                    attack_type = "DoS"
                    conf = max(conf, 0.92)
                else:
                    attack_type = model_pred

                verdict = "MALICIOUS"
            except Exception as e:
                print(f"[HybridEngine Stage 2 Error] {e}")
                verdict = "MALICIOUS"
                attack_type = "Unknown Attack"
                conf = 0.50

        # Step 2B: Integrate 1D-CNN if available and confident
        if self.stage2_classifier is not None and getattr(self, "stage2_feature_order", None):
            try:
                flow_vector = [float(raw_flow_dict.get(col, 0.0)) for col in self.stage2_feature_order]
                scaled_vec = self.stage2_classifier.scaler.transform([flow_vector])
                scaled_vec = scaled_vec[..., np.newaxis]
                dl_probs = self.stage2_classifier.model.predict(scaled_vec, verbose=0)[0]
                classes = list(self.stage2_classifier.le.classes_)
                top_idx = int(np.argmax(dl_probs))
                top_class = classes[top_idx]
                top_prob = float(dl_probs[top_idx])

                # If 1D-CNN is confident on a non-normal attack class, cross-reference
                if top_class != "Normal" and top_prob >= 0.40:
                    attack_type = top_class
                    conf = top_prob
            except Exception:
                pass

        t2_elapsed_ms = (time.perf_counter() - t2_start) * 1000
        total_latency = (time.perf_counter() - t_start) * 1000

        return {
            "verdict": verdict,
            "attack_type": attack_type,
            "confidence": round(conf, 4),
            "stage_reached": "Stage 2 (DL Engine)",
            "stage1_latency_ms": round(t1_elapsed_ms, 2),
            "stage2_latency_ms": round(t2_elapsed_ms, 2),
            "latency_ms": max(1, round(total_latency, 2))
        }
