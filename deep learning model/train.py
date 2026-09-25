import json
import random

import numpy as np
import tensorflow as tf

from sklearn.metrics import f1_score
from sklearn.utils.class_weight import (
    compute_class_weight,
)

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)

import config

from model import build_nids_model
from preprocess import load_and_preprocess


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(
    config.RANDOM_STATE
)

np.random.seed(
    config.RANDOM_STATE
)

tf.keras.utils.set_random_seed(
    config.RANDOM_STATE
)

if config.ENABLE_DETERMINISTIC_OPS:
    tf.config.experimental.enable_op_determinism()


class ValidationMacroF1(
    tf.keras.callbacks.Callback
):
    """
    Calculates macro F1 on validation data after each epoch.
    """

    def __init__(
        self,
        validation_data,
    ):
        super().__init__()

        self.validation_data = validation_data

    def on_epoch_end(
        self,
        epoch,
        logs=None,
    ):
        logs = logs or {}

        y_true = []
        y_pred = []

        for x_batch, y_batch in self.validation_data:
            probabilities = self.model.predict(
                x_batch,
                verbose=0,
            )

            predictions = np.argmax(
                probabilities,
                axis=1,
            )

            y_true.extend(
                y_batch.numpy().tolist()
            )

            y_pred.extend(
                predictions.tolist()
            )

        macro_f1 = f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        )

        logs["val_macro_f1"] = macro_f1

        print(
            f" — val_macro_f1: "
            f"{macro_f1:.4f}"
        )


def build_class_weights(
    y_train: np.ndarray,
) -> tuple[
    dict[int, float],
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Compute balanced class weights, soften and clip them.
    """

    classes = np.unique(
        y_train
    )

    raw_weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train,
    )

    if config.USE_SOFTENED_CLASS_WEIGHTS:
        final_weights = np.power(
            raw_weights,
            config.CLASS_WEIGHT_POWER,
        )
    else:
        final_weights = raw_weights

    final_weights = np.clip(
        final_weights,
        config.MIN_CLASS_WEIGHT,
        config.MAX_CLASS_WEIGHT,
    )

    class_weight_dict = {
        int(class_id): float(weight)
        for class_id, weight in zip(
            classes,
            final_weights,
        )
    }

    return (
        class_weight_dict,
        classes,
        raw_weights,
        final_weights,
    )


def tune_unknown_threshold(
    model: tf.keras.Model,
    X_val: np.ndarray,
    y_val: np.ndarray,
    label_encoder,
) -> dict:
    """
    Tune a global confidence threshold using validation data.

    The selected threshold maximizes binary attack-vs-benign F1.
    """

    probabilities = model.predict(
        X_val,
        batch_size=config.BATCH_SIZE,
        verbose=0,
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

    benign_matches = np.where(
        label_encoder.classes_ == "Benign"
    )[0]

    if len(benign_matches) == 0:
        return {
            "global_threshold": float(
                config.DEFAULT_UNKNOWN_THRESHOLD
            ),
            "metric": "not_tuned_no_benign_class",
        }

    benign_index = int(
        benign_matches[0]
    )

    y_val_attack = (
        y_val != benign_index
    ).astype(
        np.int32
    )

    best_threshold = (
        config.DEFAULT_UNKNOWN_THRESHOLD
    )

    best_score = -1.0

    for threshold in config.THRESHOLD_GRID:
        accepted_predictions = y_pred.copy()

        accepted_predictions[
            confidences < threshold
        ] = benign_index

        y_pred_attack = (
            accepted_predictions != benign_index
        ).astype(
            np.int32
        )

        score = f1_score(
            y_val_attack,
            y_pred_attack,
            zero_division=0,
        )

        if score > best_score:
            best_score = score
            best_threshold = threshold

    print(
        f"\nBest validation Unknown threshold: "
        f"{best_threshold:.2f}"
    )

    print(
        f"Validation attack-vs-benign F1: "
        f"{best_score:.4f}"
    )

    return {
        "global_threshold": float(
            best_threshold
        ),
        "metric": "binary_attack_f1",
        "validation_score": float(
            best_score
        ),
    }


def train():
    print("=" * 70)
    print(
        "TRAINING TABULAR RESIDUAL "
        "NIDS MODEL"
    )
    print("=" * 70)

    # ========================================================
    # 1. LOAD + PREPROCESS
    # ========================================================

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        label_encoder,
    ) = load_and_preprocess()

    num_classes = len(
        label_encoder.classes_
    )

    input_shape = X_train.shape[1:]

    print("\n" + "=" * 70)
    print(
        f"Input shape : {input_shape}"
    )
    print(
        f"Classes     : {num_classes}"
    )
    print(
        f"Labels      : "
        f"{list(label_encoder.classes_)}"
    )
    print("=" * 70)

    # ========================================================
    # 2. CLASS WEIGHTS
    # ========================================================

    (
        class_weight_dict,
        classes,
        raw_weights,
        final_weights,
    ) = build_class_weights(
        y_train
    )

    print(
        "\nClass weights:"
    )

    print(
        f"{'Class':15s} "
        f"{'Raw':>12s} "
        f"{'Final':>12s}"
    )

    print(
        "-" * 42
    )

    for (
        class_id,
        raw_weight,
        final_weight,
    ) in zip(
        classes,
        raw_weights,
        final_weights,
    ):
        class_name = label_encoder.classes_[
            class_id
        ]

        print(
            f"{class_name:15s} "
            f"{raw_weight:12.4f} "
            f"{final_weight:12.4f}"
        )

    # ========================================================
    # 3. TF.DATA
    # ========================================================

    print(
        "\nCreating TensorFlow datasets..."
    )

    # Dataset setup, including shuffle's seed operations, runs on the CPU.
    # Keras can still place the model and its training steps on the GPU.
    with tf.device("/CPU:0"):
        train_dataset = (
            tf.data.Dataset
            .from_tensor_slices(
                (
                    X_train,
                    y_train,
                )
            )
            .shuffle(
                buffer_size=min(
                    config.SHUFFLE_BUFFER_SIZE,
                    len(X_train),
                ),
                seed=config.RANDOM_STATE,
                reshuffle_each_iteration=True,
            )
            .batch(
                config.BATCH_SIZE,
                drop_remainder=False,
            )
            .prefetch(
                tf.data.AUTOTUNE
            )
        )

        val_dataset = (
            tf.data.Dataset
            .from_tensor_slices(
                (
                    X_val,
                    y_val,
                )
            )
            .batch(
                config.BATCH_SIZE,
                drop_remainder=False,
            )
            .prefetch(
                tf.data.AUTOTUNE
            )
        )

        test_dataset = (
            tf.data.Dataset
            .from_tensor_slices(
                (
                    X_test,
                    y_test,
                )
            )
            .batch(
                config.BATCH_SIZE,
                drop_remainder=False,
            )
            .prefetch(
                tf.data.AUTOTUNE
            )
        )

    # ========================================================
    # 4. BUILD / COMPILE
    # ========================================================

    print(
        "\nBuilding model..."
    )

    model = build_nids_model(
        input_shape=input_shape,
        num_classes=num_classes,
    )

    model.summary()

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=config.LEARNING_RATE,
    )

    if config.USE_FOCAL_LOSS:
        loss = (
            tf.keras.losses
            .SparseCategoricalFocalCrossentropy(
                gamma=config.FOCAL_GAMMA,
            )
        )

        print(
            "\nLoss: Sparse Categorical Focal Loss"
        )
    else:
        loss = (
            "sparse_categorical_crossentropy"
        )

        print(
            "\nLoss: Sparse Categorical "
            "Cross-Entropy"
        )

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=[
            tf.keras.metrics
            .SparseCategoricalAccuracy(
                name="accuracy"
            ),
        ],
    )

    # ========================================================
    # 5. CALLBACKS
    # ========================================================

    callbacks = [
        ValidationMacroF1(
            validation_data=val_dataset,
        ),
        EarlyStopping(
            monitor="val_macro_f1",
            mode="max",
            patience=config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=config.MODEL_SAVE_PATH,
            monitor="val_macro_f1",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.5,
            patience=config.LR_REDUCTION_PATIENCE,
            min_lr=config.MIN_LEARNING_RATE,
            verbose=1,
        ),
    ]

    # ========================================================
    # 6. TRAIN
    # ========================================================

    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)

    print(
        f"Maximum epochs     : "
        f"{config.EPOCHS}"
    )

    print(
        f"Batch size         : "
        f"{config.BATCH_SIZE}"
    )

    print(
        f"Training samples   : "
        f"{len(X_train):,}"
    )

    print(
        f"Validation samples : "
        f"{len(X_val):,}"
    )

    print("=" * 70)

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=config.EPOCHS,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1,
    )

    # ========================================================
    # 7. LOAD BEST MODEL
    # ========================================================

    print(
        "\nLoading best saved model..."
    )

    model = tf.keras.models.load_model(
        config.MODEL_SAVE_PATH
    )

    # ========================================================
    # 8. TUNE UNKNOWN THRESHOLD
    # ========================================================

    if config.TUNE_UNKNOWN_THRESHOLD:
        threshold_config = tune_unknown_threshold(
            model=model,
            X_val=X_val,
            y_val=y_val,
            label_encoder=label_encoder,
        )
    else:
        threshold_config = {
            "global_threshold": float(
                config.DEFAULT_UNKNOWN_THRESHOLD
            ),
            "metric": "default_configured_threshold",
        }

    with open(
        config.THRESHOLDS_SAVE_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            threshold_config,
            file,
            indent=4,
        )

    # ========================================================
    # 9. TEST LOSS / ACCURACY
    # ========================================================

    print("\n" + "=" * 70)
    print("FINAL UNSEEN TEST SET")
    print("=" * 70)

    test_loss, test_accuracy = model.evaluate(
        test_dataset,
        verbose=1,
    )

    print(
        f"\nTest loss     : "
        f"{test_loss:.6f}"
    )

    print(
        f"Test accuracy : "
        f"{test_accuracy:.4f}"
    )

    print(
        f"\nBest model saved -> "
        f"{config.MODEL_SAVE_PATH}"
    )

    return (
        model,
        history,
        X_test,
        y_test,
        label_encoder,
    )


if __name__ == "__main__":
    train()
