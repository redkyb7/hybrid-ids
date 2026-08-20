# predict.py
"""
Inference module for live / simulated packet capture.

Expected usage:
    classifier = NIDSClassifier()
    label, confidence = classifier.predict(feature_vector)

The feature_vector must be a 1-D array/list with the same 52 features
extracted from the live flow, in the same column order used during training.
"""

import numpy as np
import pickle
import tensorflow as tf

import config

# Column order expected by the model (same as training, minus label column)
FEATURE_ORDER = [
    "Destination Port", "Flow Duration", "Total Fwd Packets",
    "Total Length of Fwd Packets", "Fwd Packet Length Max",
    "Fwd Packet Length Min", "Fwd Packet Length Mean", "Fwd Packet Length Std",
    "Bwd Packet Length Max", "Bwd Packet Length Min", "Bwd Packet Length Mean",
    "Bwd Packet Length Std", "Flow Bytes/s", "Flow Packets/s",
    "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
    "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max",
    "Fwd IAT Min", "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std",
    "Bwd IAT Max", "Bwd IAT Min", "Fwd Header Length", "Bwd Header Length",
    "Fwd Packets/s", "Bwd Packets/s", "Min Packet Length", "Max Packet Length",
    "Packet Length Mean", "Packet Length Std", "Packet Length Variance",
    "FIN Flag Count", "PSH Flag Count", "ACK Flag Count",
    "Average Packet Size", "Subflow Fwd Bytes", "Init_Win_bytes_forward",
    "Init_Win_bytes_backward", "act_data_pkt_fwd", "min_seg_size_forward",
    "Active Mean", "Active Max", "Active Min", "Idle Mean", "Idle Max",
    "Idle Min",
]


class NIDSClassifier:
    def __init__(
        self,
        model_path: str = config.MODEL_SAVE_PATH,
        scaler_path: str = config.SCALER_SAVE_PATH,
        encoder_path: str = config.LABEL_ENCODER_SAVE_PATH,
    ):
        print("[NIDS] Loading model and preprocessing artifacts...")
        self.model = tf.keras.models.load_model(model_path)

        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)
        with open(encoder_path, "rb") as f:
            self.le = pickle.load(f)

        print(f"[NIDS] Ready. Classes: {list(self.le.classes_)}")

    def predict(
        self,
        feature_vector: np.ndarray | list,
        threshold: float = 0.6,
    ) -> tuple[str, float]:
        """
        Classify a single network flow.

        Parameters
        ----------
        feature_vector : array-like of shape (52,)
        threshold      : min confidence to trust prediction (else → "Unknown")

        Returns
        -------
        (label: str, confidence: float)
        """
        x = np.array(feature_vector, dtype=np.float32).reshape(1, -1)
        x = self.scaler.transform(x)
        x = x[..., np.newaxis]  # → (1, 52, 1)

        probs = self.model.predict(x, verbose=0)[0]
        class_idx = int(np.argmax(probs))
        confidence = float(probs[class_idx])

        label = self.le.classes_[class_idx] if confidence >= threshold else "Unknown"
        return label, confidence

    def predict_batch(self, feature_matrix: np.ndarray) -> list[tuple[str, float]]:
        """
        Classify a batch of flows at once (more efficient for live streams).

        Parameters
        ----------
        feature_matrix : np.ndarray of shape (N, 52)

        Returns
        -------
        List of (label, confidence) tuples
        """
        x = self.scaler.transform(feature_matrix.astype(np.float32))
        x = x[..., np.newaxis]  # → (N, 52, 1)

        probs = self.model.predict(x, batch_size=256, verbose=0)
        indices = np.argmax(probs, axis=1)
        confidences = probs[np.arange(len(probs)), indices]

        return [
            (self.le.classes_[i], float(c))
            for i, c in zip(indices, confidences)
        ]


# ── Quick smoke test ──────────────────────────────────────────────────────
if __name__ == "__main__":
    classifier = NIDSClassifier()

    # Simulate a random flow (replace with real CICFlowMeter output)
    dummy_flow = np.random.randn(52)
    label, conf = classifier.predict(dummy_flow)
    print(f"Prediction: {label}  (confidence: {conf:.4f})")