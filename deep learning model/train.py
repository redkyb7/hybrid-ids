# train.py

import random

import numpy as np
import tensorflow as tf

from sklearn.utils.class_weight import (
    compute_class_weight,
)

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)

import config

from preprocess import load_and_preprocess
from model import build_cnn


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


def train():

    print("=" * 70)
    print(
        "TRAINING 1D-CNN NETWORK "
        "INTRUSION DETECTION MODEL"
    )
    print("=" * 70)


    # ========================================================
    # 1. LOAD + PREPROCESS DATA
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
    # 2. COMPUTE RAW BALANCED CLASS WEIGHTS
    # ========================================================

    print(
        "\nCalculating class weights..."
    )


    classes = np.unique(
        y_train
    )


    raw_weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train,
    )


    # ========================================================
    # 3. SOFTEN EXTREME CLASS WEIGHTS
    # ========================================================
    #
    # Previous weights were approximately:
    #
    # Botnet       = 184.8
    # Brute Force  = 39.4
    # DDoS         = 2.81
    # DoS          = 1.86
    # Normal       = 0.17
    # Port Scan    = 3.97
    # Web Attack   = 168.1
    #
    # These weights caused very high recall but extremely
    # poor precision for rare classes such as Botnet.
    #
    # Square-root compression gives roughly:
    #
    # Botnet       = 13.6
    # Brute Force  = 6.3
    # DDoS         = 1.7
    # DoS          = 1.4
    # Normal       = 0.4
    # Port Scan    = 2.0
    # Web Attack   = 13.0
    #
    # This still prioritises minority attacks without allowing
    # individual rare samples to dominate the loss function.
    # ========================================================

    if config.USE_SOFTENED_CLASS_WEIGHTS:

        final_weights = np.sqrt(
            raw_weights
        )

        print(
            "\nUsing SQRT-softened "
            "class weights."
        )

    else:

        final_weights = raw_weights

        print(
            "\nUsing raw balanced "
            "class weights."
        )


    class_weight_dict = {

        int(class_id): float(weight)

        for class_id, weight
        in zip(
            classes,
            final_weights,
        )
    }


    # ========================================================
    # DISPLAY RAW + FINAL WEIGHTS
    # ========================================================

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

        class_name = (
            label_encoder.classes_[
                class_id
            ]
        )

        print(
            f"{class_name:15s} "
            f"{raw_weight:12.4f} "
            f"{final_weight:12.4f}"
        )


    # ========================================================
    # 4. CREATE TF.DATA PIPELINES
    # ========================================================

    print(
        "\nCreating TensorFlow "
        "data pipeline..."
    )


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
    # 5. BUILD MODEL
    # ========================================================

    print(
        "\nBuilding CNN..."
    )


    model = build_cnn(
        input_shape=input_shape,
        num_classes=num_classes,
    )


    model.summary()


    # ========================================================
    # 6. COMPILE MODEL
    # ========================================================

    optimizer = (
        tf.keras.optimizers.Adam(
            learning_rate=(
                config.LEARNING_RATE
            )
        )
    )


    model.compile(
        optimizer=optimizer,
        loss=(
            "sparse_categorical_crossentropy"
        ),
        metrics=[
            "accuracy",
        ],
    )


    # ========================================================
    # 7. CALLBACKS
    # ========================================================

    callbacks = [

        # ----------------------------------------------------
        # EARLY STOPPING
        # ----------------------------------------------------

        EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=(
                config
                .EARLY_STOPPING_PATIENCE
            ),
            restore_best_weights=True,
            verbose=1,
        ),


        # ----------------------------------------------------
        # SAVE BEST MODEL
        # ----------------------------------------------------

        ModelCheckpoint(
            filepath=(
                config.MODEL_SAVE_PATH
            ),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1,
        ),


        # ----------------------------------------------------
        # REDUCE LEARNING RATE
        # ----------------------------------------------------

        ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.5,
            patience=(
                config
                .LR_REDUCTION_PATIENCE
            ),
            min_lr=(
                config
                .MIN_LEARNING_RATE
            ),
            verbose=1,
        ),
    ]


    # ========================================================
    # 8. TRAIN MODEL
    # ========================================================

    training_steps = int(
        np.ceil(
            len(X_train)
            / config.BATCH_SIZE
        )
    )


    validation_steps = int(
        np.ceil(
            len(X_val)
            / config.BATCH_SIZE
        )
    )


    print("\n" + "=" * 70)

    print(
        "STARTING TRAINING"
    )

    print("=" * 70)

    print(
        f"Maximum epochs   : "
        f"{config.EPOCHS}"
    )

    print(
        f"Batch size       : "
        f"{config.BATCH_SIZE}"
    )

    print(
        f"Training samples : "
        f"{len(X_train):,}"
    )

    print(
        f"Training steps   : "
        f"{training_steps:,}"
    )

    print(
        f"Validation samples: "
        f"{len(X_val):,}"
    )

    print(
        f"Validation steps : "
        f"{validation_steps:,}"
    )

    print("=" * 70)


    history = model.fit(

        train_dataset,

        validation_data=(
            val_dataset
        ),

        epochs=(
            config.EPOCHS
        ),

        class_weight=(
            class_weight_dict
        ),

        callbacks=callbacks,

        verbose=1,
    )


    # ========================================================
    # 9. LOAD BEST MODEL
    # ========================================================

    print(
        "\nLoading best saved model..."
    )


    model = (
        tf.keras.models.load_model(
            config.MODEL_SAVE_PATH
        )
    )


    # ========================================================
    # 10. FINAL UNSEEN TEST SET
    # ========================================================

    print("\n" + "=" * 70)

    print(
        "FINAL UNSEEN TEST SET"
    )

    print("=" * 70)


    test_loss, test_accuracy = (
        model.evaluate(
            test_dataset,
            verbose=1,
        )
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


    # ========================================================
    # RETURN
    # ========================================================

    return (
        model,
        history,
        X_test,
        y_test,
        label_encoder,
    )


if __name__ == "__main__":
    train()