# train.py
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard,
)
from sklearn.utils.class_weight import compute_class_weight

import config
from preprocess import load_and_preprocess
from model import build_cnn


def train():
    # ── Load data ─────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test, le = load_and_preprocess()

    num_classes = len(le.classes_)
    input_shape = X_train.shape[1:]  # (n_features, 1)

    print(f"\nInput shape : {input_shape}")
    print(f"Num classes : {num_classes} -> {list(le.classes_)}\n")

    # ── Class weights (extra safety on top of resampling) ─────────────────
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train,
    )
    class_weight_dict = dict(enumerate(class_weights))

    # ── Build & compile ───────────────────────────────────────────────────
    model = build_cnn(input_shape, num_classes)
    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # ── Callbacks ─────────────────────────────────────────────────────────
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=config.MODEL_SAVE_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
        TensorBoard(log_dir="logs/"),
    ]

    # ── Train ─────────────────────────────────────────────────────────────
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1,
    )

    print(f"\nModel saved -> {config.MODEL_SAVE_PATH}")
    return model, history, X_test, y_test, le


if __name__ == "__main__":
    train()