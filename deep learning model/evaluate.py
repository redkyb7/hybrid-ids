# evaluate.py

import json
import os
import pickle

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


def evaluate():

    print("=" * 70)
    print("EVALUATING PRE-TRAINED 1D-CNN NIDS MODEL")
    print("=" * 70)


    # ========================================================
    # 1. CHECK REQUIRED FILES
    # ========================================================

    required_files = [
        config.MODEL_SAVE_PATH,
        config.LABEL_ENCODER_SAVE_PATH,
        config.X_TEST_SAVE_PATH,
        config.Y_TEST_SAVE_PATH,
    ]


    for path in required_files:

        if not os.path.exists(path):

            raise FileNotFoundError(
                "\nRequired file does not exist:\n"
                f"{path}\n\n"
                "Run train.py first."
            )


    # ========================================================
    # 2. LOAD MODEL
    # ========================================================

    print(
        f"\n[*] Loading model:\n"
        f"    {config.MODEL_SAVE_PATH}"
    )

    model = tf.keras.models.load_model(
        config.MODEL_SAVE_PATH
    )


    # ========================================================
    # 3. LOAD LABEL ENCODER
    # ========================================================

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


    # ========================================================
    # 4. LOAD EXACT HELD-OUT TEST SET
    # ========================================================

    print(
        "\n[*] Loading unseen test set..."
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


    # ========================================================
    # 5. MODEL INFERENCE
    # ========================================================

    print(
        "\n[*] Running inference..."
    )


    probabilities = model.predict(
        X_test,
        batch_size=config.BATCH_SIZE,
        verbose=1,
    )


    y_pred = np.argmax(
        probabilities,
        axis=1,
    )


    # ========================================================
    # 6. ACCURACY
    # ========================================================

    accuracy = accuracy_score(
        y_test,
        y_pred,
    )


    # ========================================================
    # 7. MACRO METRICS
    # ========================================================

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


    # ========================================================
    # 8. WEIGHTED METRICS
    # ========================================================

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


    # ========================================================
    # 9. OVERALL RESULTS
    # ========================================================

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


    # ========================================================
    # 10. CLASSIFICATION REPORT
    # ========================================================

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


    # ========================================================
    # 11. SAVE CLASSIFICATION REPORT
    # ========================================================

    report_dict = classification_report(
        y_test,
        y_pred,
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )


    report_path = os.path.join(
        config.SAVED_MODEL_DIR,
        "classification_report.json",
    )


    with open(
        report_path,
        "w",
    ) as file:

        json.dump(
            report_dict,
            file,
            indent=4,
        )


    # ========================================================
    # 12. CONFUSION MATRIX
    # ========================================================

    cm = confusion_matrix(
        y_test,
        y_pred,
    )


    figure, axis = plt.subplots(
        figsize=(12, 10)
    )


    image = axis.imshow(
        cm,
        interpolation="nearest",
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
        title=(
            "Confusion Matrix - "
            "1D CNN NIDS"
        ),
    )


    plt.setp(
        axis.get_xticklabels(),
        rotation=45,
        ha="right",
        rotation_mode="anchor",
    )


    # Add values inside matrix
    threshold = (
        cm.max() / 2
        if cm.size
        else 0
    )


    for row in range(
        cm.shape[0]
    ):

        for column in range(
            cm.shape[1]
        ):

            value = cm[
                row,
                column
            ]

            axis.text(
                column,
                row,
                f"{value:,}",
                ha="center",
                va="center",
            )


    figure.tight_layout()


    confusion_matrix_path = (
        os.path.join(
            config.SAVED_MODEL_DIR,
            "confusion_matrix.png",
        )
    )


    figure.savefig(
        confusion_matrix_path,
        dpi=200,
        bbox_inches="tight",
    )


    plt.close(
        figure
    )


    # ========================================================
    # 13. SAVE SUMMARY METRICS
    # ========================================================

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
    }


    metrics_path = os.path.join(
        config.SAVED_MODEL_DIR,
        "evaluation_metrics.json",
    )


    with open(
        metrics_path,
        "w",
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4,
        )


    # ========================================================
    # 14. OUTPUT LOCATIONS
    # ========================================================

    print("\n" + "=" * 70)
    print("FILES SAVED")
    print("=" * 70)


    print(
        f"Classification report -> "
        f"{report_path}"
    )

    print(
        f"Evaluation metrics     -> "
        f"{metrics_path}"
    )

    print(
        f"Confusion matrix       -> "
        f"{confusion_matrix_path}"
    )


    print("=" * 70)


    return metrics


if __name__ == "__main__":
    evaluate()