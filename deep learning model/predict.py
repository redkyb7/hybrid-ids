import json
import os
import pickle

import numpy as np
import tensorflow as tf

if __package__:
    from . import config
else:
    import config


class NIDSClassifier:
    """
    Inference interface for the trained NIDS model.

    Accepted inputs:

    - Dictionary containing all saved feature names.
    - List / NumPy array matching the exact saved feature order.
    """

    def __init__(
        self,
        model_path: str = config.MODEL_SAVE_PATH,
        scaler_path: str = config.SCALER_SAVE_PATH,
        encoder_path: str = config.LABEL_ENCODER_SAVE_PATH,
        feature_names_path: str = config.FEATURE_NAMES_SAVE_PATH,
        metadata_path: str = config.METADATA_SAVE_PATH,
        thresholds_path: str = config.THRESHOLDS_SAVE_PATH,
    ):
        print(
            "[NIDS] Loading model and artifacts..."
        )

        required_files = [
            model_path,
            scaler_path,
            encoder_path,
            feature_names_path,
        ]

        for path in required_files:
            if not os.path.exists(path):
                raise FileNotFoundError(
                    "Required artifact does not exist:\n"
                    f"{path}\n\n"
                    "Run train.py first."
                )

        self.model = tf.keras.models.load_model(
            model_path
        )

        with open(
            scaler_path,
            "rb",
        ) as file:
            self.scaler = pickle.load(
                file
            )

        with open(
            encoder_path,
            "rb",
        ) as file:
            self.label_encoder = pickle.load(
                file
            )

        with open(
            feature_names_path,
            "rb",
        ) as file:
            self.feature_names = pickle.load(
                file
            )

        self.num_features = len(
            self.feature_names
        )

        self.metadata = {}

        if os.path.exists(
            metadata_path
        ):
            with open(
                metadata_path,
                "r",
                encoding="utf-8",
            ) as file:
                self.metadata = json.load(
                    file
                )

        self.threshold_config = {
            "global_threshold": (
                config.DEFAULT_UNKNOWN_THRESHOLD
            )
        }

        if os.path.exists(
            thresholds_path
        ):
            with open(
                thresholds_path,
                "r",
                encoding="utf-8",
            ) as file:
                self.threshold_config = json.load(
                    file
                )

        self.default_threshold = float(
            self.threshold_config.get(
                "global_threshold",
                config.DEFAULT_UNKNOWN_THRESHOLD,
            )
        )

        self._validate_artifacts()

        # Compatibility alias.
        self.le = self.label_encoder

        print(
            "[NIDS] Ready."
        )

        print(
            f"[NIDS] Features: "
            f"{self.num_features}"
        )

        print(
            f"[NIDS] Classes: "
            f"{list(self.label_encoder.classes_)}"
        )

        print(
            f"[NIDS] Default threshold: "
            f"{self.default_threshold:.2f}"
        )

    @property
    def feature_order(
        self,
    ) -> list[str]:
        """
        Exact feature order required for list/array input.
        """

        return list(
            self.feature_names
        )

    def _validate_artifacts(
        self,
    ):
        """
        Ensure model, scaler, labels and schema match.
        """

        model_feature_count = self.model.input_shape[1]

        if model_feature_count != self.num_features:
            raise ValueError(
                "Model / feature schema mismatch. "
                f"Model expects {model_feature_count} features, "
                f"but feature_names.pkl contains "
                f"{self.num_features}."
            )

        model_class_count = self.model.output_shape[-1]

        encoder_class_count = len(
            self.label_encoder.classes_
        )

        if model_class_count != encoder_class_count:
            raise ValueError(
                "Model / label encoder mismatch. "
                f"Model outputs {model_class_count} classes, "
                f"but label encoder contains "
                f"{encoder_class_count}."
            )

        scaler_feature_count = getattr(
            self.scaler,
            "n_features_in_",
            None,
        )

        if (
            scaler_feature_count is not None
            and scaler_feature_count
            != self.num_features
        ):
            raise ValueError(
                "Scaler / feature schema mismatch."
            )

    def _clip_raw_features(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """
        Apply saved train-derived clipping bounds.
        """

        clipping = self.metadata.get(
            "percentile_clipping",
            {},
        )

        if not clipping.get(
            "enabled",
            False,
        ):
            return X

        lower_bounds = clipping.get(
            "lower_bounds"
        )

        upper_bounds = clipping.get(
            "upper_bounds"
        )

        if (
            lower_bounds is None
            or upper_bounds is None
        ):
            return X

        return np.clip(
            X,
            np.asarray(
                lower_bounds,
                dtype=np.float32,
            ),
            np.asarray(
                upper_bounds,
                dtype=np.float32,
            ),
        )

    def _prepare_feature_matrix(
        self,
        feature_matrix,
    ) -> np.ndarray:
        """
        Validate, clip, scale and reshape flow features.
        """

        X = np.asarray(
            feature_matrix,
            dtype=np.float32,
        )

        if X.ndim != 2:
            raise ValueError(
                "feature_matrix must have shape "
                "(N, number_of_features)."
            )

        if X.shape[1] != self.num_features:
            raise ValueError(
                f"Expected {self.num_features} features, "
                f"received {X.shape[1]}."
            )

        if not np.all(
            np.isfinite(X)
        ):
            raise ValueError(
                "Input contains NaN or infinite values."
            )

        X = self._clip_raw_features(
            X
        )

        X = self.scaler.transform(
            X
        ).astype(
            np.float32
        )

        return X[
            ..., np.newaxis
        ]

    def _prepare_single_flow(
        self,
        feature_vector,
    ) -> np.ndarray:
        """
        Prepare one feature dictionary or ordered vector.
        """

        if isinstance(
            feature_vector,
            dict,
        ):
            missing_features = [
                feature
                for feature in self.feature_names
                if feature not in feature_vector
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
                for feature in self.feature_names
            ]
        else:
            values = feature_vector

        X = np.asarray(
            values,
            dtype=np.float32,
        ).reshape(
            1,
            -1,
        )

        return self._prepare_feature_matrix(
            X
        )

    def _resolve_threshold(
        self,
        threshold: float | None,
    ) -> float:
        """
        Use supplied threshold or saved validation-tuned default.
        """

        if threshold is None:
            threshold = self.default_threshold

        if not 0.0 <= threshold <= 1.0:
            raise ValueError(
                "threshold must be between 0 and 1."
            )

        return float(
            threshold
        )

    def _predict_single_probabilities(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """Use the loaded model directly for low-latency single-flow inference."""
        return self.model(
            X,
            training=False,
        ).numpy()[0]

    def predict(
        self,
        feature_vector,
        threshold: float | None = None,
    ) -> tuple[str, float]:
        """
        Predict one flow.

        Returns:
            (label, confidence)

        If confidence is below the threshold, label is "Unknown".
        """

        threshold = self._resolve_threshold(
            threshold
        )

        X = self._prepare_single_flow(
            feature_vector
        )

        probabilities = self._predict_single_probabilities(
            X
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
            label = self.label_encoder.classes_[
                class_index
            ]

        return (
            label,
            confidence,
        )

    def predict_detailed(
        self,
        feature_vector,
        threshold: float | None = None,
    ) -> dict:
        """
        Predict one flow and return all class probabilities.
        """

        threshold = self._resolve_threshold(
            threshold
        )

        X = self._prepare_single_flow(
            feature_vector
        )

        probabilities = self._predict_single_probabilities(
            X
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

        predicted_class = self.label_encoder.classes_[
            class_index
        ]

        label = (
            predicted_class
            if confidence >= threshold
            else "Unknown"
        )

        probability_map = {
            class_name: float(probability)
            for class_name, probability in zip(
                self.label_encoder.classes_,
                probabilities,
            )
        }

        return {
            "label": label,
            "predicted_class": predicted_class,
            "confidence": confidence,
            "threshold": threshold,
            "probabilities": probability_map,
        }

    def predict_batch(
        self,
        feature_matrix,
        threshold: float | None = None,
    ) -> list[tuple[str, float]]:
        """
        Predict multiple flows.

        feature_matrix shape:
            (number_of_flows, number_of_features)
        """

        threshold = self._resolve_threshold(
            threshold
        )

        X = self._prepare_feature_matrix(
            feature_matrix
        )

        probabilities = self.model.predict(
            X,
            batch_size=1024,
            verbose=0,
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

        for class_index, confidence in zip(
            class_indices,
            confidences,
        ):
            confidence = float(
                confidence
            )

            if confidence < threshold:
                label = "Unknown"
            else:
                label = self.label_encoder.classes_[
                    class_index
                ]

            results.append(
                (
                    label,
                    confidence,
                )
            )

        return results

    def show_feature_order(
        self,
    ):
        """
        Display required feature names and list-input ordering.
        """

        print(
            "\nExpected feature order:"
        )

        for index, feature in enumerate(
            self.feature_names,
            start=1,
        ):
            print(
                f"{index:02d}. {feature}"
            )


if __name__ == "__main__":
    classifier = NIDSClassifier()

    classifier.show_feature_order()

    dummy_flow = np.zeros(
        classifier.num_features,
        dtype=np.float32,
    )

    result = classifier.predict_detailed(
        dummy_flow
    )

    print(
        "\nSmoke test:"
    )

    print(
        f"Label      : "
        f"{result['label']}"
    )

    print(
        f"Confidence : "
        f"{result['confidence']:.4f}"
    )
