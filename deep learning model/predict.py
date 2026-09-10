# predict.py

import pickle

import numpy as np
import tensorflow as tf

import config


# Feature order exported for external consumers (e.g. HybridIDSEngine)
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
    """
    Inference class for the trained
    1D-CNN network intrusion detection model.
    """

    def __init__(
        self,
        model_path: str = config.MODEL_SAVE_PATH,
        scaler_path: str = config.SCALER_SAVE_PATH,
        encoder_path: str = config.LABEL_ENCODER_SAVE_PATH,
        feature_names_path: str = config.FEATURE_NAMES_SAVE_PATH,
    ):

        print(
            "[NIDS] Loading model "
            "and preprocessing artifacts..."
        )


        # ====================================================
        # LOAD MODEL
        # ====================================================

        self.model = (
            tf.keras.models.load_model(
                model_path
            )
        )


        # ====================================================
        # LOAD SCALER
        # ====================================================

        with open(
            scaler_path,
            "rb",
        ) as file:

            self.scaler = pickle.load(
                file
            )


        # ====================================================
        # LOAD LABEL ENCODER
        # ====================================================

        with open(
            encoder_path,
            "rb",
        ) as file:

            self.label_encoder = (
                pickle.load(
                    file
                )
            )


        # ====================================================
        # LOAD TRAINING FEATURE ORDER
        # ====================================================

        with open(
            feature_names_path,
            "rb",
        ) as file:

            self.feature_names = (
                pickle.load(
                    file
                )
            )


        self.num_features = len(
            self.feature_names
        )


        self.le = self.label_encoder

        print(
            f"[NIDS] Ready."
        )

        print(
            f"[NIDS] Features: "
            f"{self.num_features}"
        )

        print(
            f"[NIDS] Classes: "
            f"{list(self.label_encoder.classes_)}"
        )


    # ========================================================
    # PREPROCESS ONE FLOW
    # ========================================================

    def _prepare_single_flow(
        self,
        feature_vector,
    ):

        # Dictionary input
        if isinstance(
            feature_vector,
            dict,
        ):

            missing_features = [
                feature
                for feature
                in self.feature_names
                if feature
                not in feature_vector
            ]

            if missing_features:

                raise ValueError(
                    "Missing required features:\n"
                    + "\n".join(
                        missing_features
                    )
                )

            values = [
                feature_vector[
                    feature
                ]
                for feature
                in self.feature_names
            ]


        # List / NumPy array input
        else:

            values = feature_vector


        x = np.asarray(
            values,
            dtype=np.float32,
        ).reshape(
            1,
            -1,
        )


        # Validate feature count
        if x.shape[1] != self.num_features:

            raise ValueError(
                f"Expected "
                f"{self.num_features} features, "
                f"but received "
                f"{x.shape[1]}."
            )


        # Validate finite values
        if not np.all(
            np.isfinite(x)
        ):

            raise ValueError(
                "Input contains NaN or "
                "infinite values."
            )


        # Scale
        x = self.scaler.transform(
            x
        ).astype(
            np.float32
        )


        # CNN format:
        # (1, features, 1)
        x = x[
            ..., np.newaxis
        ]


        return x


    # ========================================================
    # PREDICT SINGLE FLOW
    # ========================================================

    def predict(
        self,
        feature_vector,
        threshold: float = 0.60,
    ) -> tuple[str, float]:
        """
        Classify one network flow.

        Parameters
        ----------
        feature_vector:
            List/array containing all feature
            values in training order,

            OR

            dictionary:
                {
                    feature_name: value
                }

        threshold:
            Minimum confidence required.
            Predictions below this threshold
            return "Unknown".

        Returns
        -------
        tuple:
            (predicted_label, confidence)
        """

        if not (
            0.0
            <= threshold
            <= 1.0
        ):

            raise ValueError(
                "threshold must be "
                "between 0 and 1."
            )


        x = self._prepare_single_flow(
            feature_vector
        )


        probabilities = (
            self.model.predict(
                x,
                verbose=0,
            )[0]
        )


        class_index = int(
            np.argmax(
                probabilities
            )
        )


        confidence = float(
            probabilities[
                class_index
            ]
        )


        if confidence < threshold:

            label = "Unknown"

        else:

            label = (
                self
                .label_encoder
                .classes_[
                    class_index
                ]
            )


        return (
            label,
            confidence,
        )


    # ========================================================
    # PREDICT BATCH
    # ========================================================

    def predict_batch(
        self,
        feature_matrix,
        threshold: float = 0.60,
    ):
        """
        Classify multiple network flows.

        Input:
            shape (N, number_of_features)
        """

        X = np.asarray(
            feature_matrix,
            dtype=np.float32,
        )


        if X.ndim != 2:

            raise ValueError(
                "feature_matrix must have "
                "shape (N, features)."
            )


        if X.shape[1] != self.num_features:

            raise ValueError(
                f"Expected "
                f"{self.num_features} features, "
                f"received {X.shape[1]}."
            )


        if not np.all(
            np.isfinite(X)
        ):

            raise ValueError(
                "Input contains NaN or "
                "infinite values."
            )


        X = self.scaler.transform(
            X
        ).astype(
            np.float32
        )


        X = X[
            ..., np.newaxis
        ]


        probabilities = (
            self.model.predict(
                X,
                batch_size=1024,
                verbose=0,
            )
        )


        class_indices = np.argmax(
            probabilities,
            axis=1,
        )


        confidences = probabilities[
            np.arange(
                len(probabilities)
            ),
            class_indices,
        ]


        results = []


        for (
            class_index,
            confidence,
        ) in zip(
            class_indices,
            confidences,
        ):

            confidence = float(
                confidence
            )


            if confidence < threshold:

                label = "Unknown"

            else:

                label = (
                    self
                    .label_encoder
                    .classes_[
                        class_index
                    ]
                )


            results.append(
                (
                    label,
                    confidence,
                )
            )


        return results


    # ========================================================
    # PRINT FEATURE ORDER
    # ========================================================

    def show_feature_order(
        self
    ):

        print(
            "\nExpected feature order:"
        )


        for index, feature in enumerate(
            self.feature_names,
            start=1,
        ):

            print(
                f"{index:02d}. "
                f"{feature}"
            )


# ============================================================
# SMOKE TEST
# ============================================================

if __name__ == "__main__":

    classifier = NIDSClassifier()

    classifier.show_feature_order()

    # Random values are ONLY a software
    # smoke test, not a meaningful flow.
    dummy_flow = np.zeros(
        classifier.num_features,
        dtype=np.float32,
    )

    label, confidence = (
        classifier.predict(
            dummy_flow,
            threshold=0.60,
        )
    )

    print(
        f"\nSmoke test prediction: "
        f"{label}"
    )

    print(
        f"Confidence: "
        f"{confidence:.4f}"
    )