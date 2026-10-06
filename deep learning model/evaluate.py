import json
import os
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

import config


def save_confusion_matrix(
    matrix: np.ndarray,
    classes: list[str],
    title: str,
    output_path: str,
    normalized: bool,
):
    """
    Save a count or true-class-normalized confusion matrix.
    """

    figure, axis = plt.subplots(
        figsize=(12, 10)
    )

    image = axis.imshow(
        matrix,
        interpolation="nearest",
        cmap="Blues",
    )

    figure.colorbar(
        image,
        ax=axis,
    )

    axis.set(
        xticks=np.arange(
            len(classes)
        ),
        yticks=np.arange(
            len(classes)
        ),
        xticklabels=classes,
        yticklabels=classes,
        ylabel="True Class",
        xlabel="Predicted Class",
        title=title,
    )

    plt.setp(
        axis.get_xticklabels(),
        rotation=45,
        ha="right",
        rotation_mode="anchor",
    )

    threshold = (
        matrix.max() / 2
        if matrix.size
        else 0
    )

    for row in range(
        matrix.shape[0]
    ):
        for column in range(
            matrix.shape[1]
        ):
            value = matrix[
                row,
                column
            ]

            text = (
                f"{value:.2f}"
                if normalized
                else f"{int(value):,}"
            )

            text_color = (
                "white"
                if value > threshold
                else "black"
            )

            axis.text(
                column,
                row,
                text,
                ha="center",
                va="center",
                color=text_color,
                fontsize=8,
            )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


def evaluate():
    print("=" * 70)
    print(
        "EVALUATING PRE-TRAINED "
        "NIDS MODEL"
    )
    print("=" * 70)

    required_files = [
        config.MODEL_SAVE_PATH,
        config.LABEL_ENCODER_SAVE_PATH,
        config.X_TEST_SAVE_PATH,
        config.Y_TEST_SAVE_PATH,
    ]

    for path in required_files:
        if not os.path.exists(path):
            raise FileNotFoundError(
                "Required file does not exist:\n"
                f"{path}\n\n"
                "Run train.py first."
            )

    print(
        f"\n[*] Loading model:\n"
        f"    {config.MODEL_SAVE_PATH}"
    )

    model = tf.keras.models.load_model(
        config.MODEL_SAVE_PATH
    )

    with open(
        config.LABEL_ENCODER_SAVE_PATH,
        "rb",
    ) as file:
        label_encoder = pickle.load(
            file
        )

    classes = list(
        label_encoder.classes_
    )

    print(
        f"\n[+] Classes ({len(classes)}):"
    )

    for index, class_name in enumerate(
        classes
    ):
        print(
            f"    {index}: "
            f"{class_name}"
        )

    print(
        "\n[*] Loading exact unseen test set..."
    )

    X_test = np.load(
        config.X_TEST_SAVE_PATH,
        mmap_mode="r",
    )

    y_test = np.load(
        config.Y_TEST_SAVE_PATH,
    )

    print(
        f"[+] Test samples : "
        f"{len(y_test):,}"
    )

    print(
        f"[+] Input shape  : "
        f"{X_test.shape}"
    )

    print(
        "\n[*] Running inference..."
    )

    start_time = time.perf_counter()

    probabilities = model.predict(
        X_test,
        batch_size=config.BATCH_SIZE,
        verbose=1,
    )

    inference_seconds = (
        time.perf_counter()
        - start_time
    )

    y_pred = np.argmax(
        probabilities,
        axis=1,
    )

    confidences = probabilities[
        np.arange(
            len(probabilities)
        ),
        y_pred,
    ]

    throughput = (
        len(y_test) / inference_seconds
        if inference_seconds > 0
        else 0.0
    )

    accuracy = accuracy_score(
        y_test,
        y_pred,
    )

    (
        macro_precision,
        macro_recall,
        macro_f1,
        _,
    ) = precision_recall_fscore_support(
        y_test,
        y_pred,
        average="macro",
        zero_division=0,
    )

    (
        weighted_precision,
        weighted_recall,
        weighted_f1,
        _,
    ) = precision_recall_fscore_support(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    benign_matches = np.where(
        label_encoder.classes_ == "Benign"
    )[0]

    binary_metrics = None

    if len(benign_matches) > 0:
        benign_index = int(
            benign_matches[0]
        )

        y_test_attack = (
            y_test != benign_index
        ).astype(
            np.int32
        )

        y_pred_attack = (
            y_pred != benign_index
        ).astype(
            np.int32
        )

        (
            attack_precision,
            attack_recall,
            attack_f1,
            _,
        ) = precision_recall_fscore_support(
            y_test_attack,
            y_pred_attack,
            average="binary",
            zero_division=0,
        )

        true_benign = y_test == benign_index

        false_positive_count = int(
            np.sum(
                (y_pred != benign_index)
                & true_benign
            )
        )

        benign_count = int(
            np.sum(true_benign)
        )

        false_positive_rate = (
            false_positive_count / benign_count
            if benign_count > 0
            else 0.0
        )

        binary_metrics = {
            "attack_precision": float(
                attack_precision
            ),
            "attack_recall": float(
                attack_recall
            ),
            "attack_f1": float(
                attack_f1
            ),
            "false_positive_count": int(
                false_positive_count
            ),
            "benign_samples": int(
                benign_count
            ),
            "false_positive_rate": float(
                false_positive_rate
            ),
        }

    print("\n" + "=" * 70)
    print("OVERALL TEST RESULTS")
    print("=" * 70)

    print(
        f"Accuracy           : "
        f"{accuracy:.4f}"
    )

    print()

    print(
        f"Macro Precision    : "
        f"{macro_precision:.4f}"
    )

    print(
        f"Macro Recall       : "
        f"{macro_recall:.4f}"
    )

    print(
        f"Macro F1           : "
        f"{macro_f1:.4f}"
    )

    print()

    print(
        f"Weighted Precision : "
        f"{weighted_precision:.4f}"
    )

    print(
        f"Weighted Recall    : "
        f"{weighted_recall:.4f}"
    )

    print(
        f"Weighted F1        : "
        f"{weighted_f1:.4f}"
    )

    print()

    print(
        f"Inference time     : "
        f"{inference_seconds:.2f} seconds"
    )

    print(
        f"Throughput         : "
        f"{throughput:,.2f} flows/second"
    )

    print(
        f"Mean confidence    : "
        f"{np.mean(confidences):.4f}"
    )

    if binary_metrics is not None:
        print("\n" + "=" * 70)
        print(
            "BINARY ATTACK-VS-BENIGN RESULTS"
        )
        print("=" * 70)

        print(
            f"Attack Precision   : "
            f"{binary_metrics['attack_precision']:.4f}"
        )

        print(
            f"Attack Recall      : "
            f"{binary_metrics['attack_recall']:.4f}"
        )

        print(
            f"Attack F1          : "
            f"{binary_metrics['attack_f1']:.4f}"
        )

        print(
            f"False Positive Rate: "
            f"{binary_metrics['false_positive_rate']:.6f}"
        )

        print(
            f"False Positives    : "
            f"{binary_metrics['false_positive_count']:,}"
        )

    print("\n" + "=" * 70)
    print("PER-CLASS CLASSIFICATION REPORT")
    print("=" * 70)

    report_text = classification_report(
        y_test,
        y_pred,
        target_names=classes,
        digits=4,
        zero_division=0,
    )

    print(
        report_text
    )

    report_dict = classification_report(
        y_test,
        y_pred,
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )

    with open(
        config.CLASSIFICATION_REPORT_SAVE_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report_dict,
            file,
            indent=4,
        )

    count_matrix = confusion_matrix(
        y_test,
        y_pred,
    )

    normalized_matrix = confusion_matrix(
        y_test,
        y_pred,
        normalize="true",
    )

    save_confusion_matrix(
        matrix=count_matrix,
        classes=classes,
        title=(
            "Confusion Matrix - "
            "NIDS (Counts)"
        ),
        output_path=(
            config.CONFUSION_MATRIX_SAVE_PATH
        ),
        normalized=False,
    )

    save_confusion_matrix(
        matrix=normalized_matrix,
        classes=classes,
        title=(
            "Confusion Matrix - "
            "NIDS (True-Class Normalized)"
        ),
        output_path=(
            config
            .NORMALIZED_CONFUSION_MATRIX_SAVE_PATH
        ),
        normalized=True,
    )

    metrics = {
        "accuracy": float(
            accuracy
        ),
        "macro_precision": float(
            macro_precision
        ),
        "macro_recall": float(
            macro_recall
        ),
        "macro_f1": float(
            macro_f1
        ),
        "weighted_precision": float(
            weighted_precision
        ),
        "weighted_recall": float(
            weighted_recall
        ),
        "weighted_f1": float(
            weighted_f1
        ),
        "test_samples": int(
            len(y_test)
        ),
        "inference_seconds": float(
            inference_seconds
        ),
        "throughput_flows_per_second": float(
            throughput
        ),
        "mean_prediction_confidence": float(
            np.mean(confidences)
        ),
        "binary_attack_vs_benign": binary_metrics,
    }

    with open(
        config.METRICS_SAVE_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            indent=4,
        )

    print("\n" + "=" * 70)
    print("FILES SAVED")
    print("=" * 70)

    print(
        f"Classification report -> "
        f"{config.CLASSIFICATION_REPORT_SAVE_PATH}"
    )

    print(
        f"Evaluation metrics   -> "
        f"{config.METRICS_SAVE_PATH}"
    )

    print(
        f"Confusion matrix     -> "
        f"{config.CONFUSION_MATRIX_SAVE_PATH}"
    )

    print(
        f"Normalized matrix    -> "
        f"{config.NORMALIZED_CONFUSION_MATRIX_SAVE_PATH}"
    )

    print("=" * 70)

    return metrics


if __name__ == "__main__":
    evaluate()