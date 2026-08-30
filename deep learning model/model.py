# model.py
import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers


def build_cnn(input_shape: tuple, num_classes: int) -> Model:
    """
    1D-CNN for tabular network-flow classification.

    Architecture:
      Input (features, 1)
        → Conv block × 3  (increasing filters, residual-style)
        → GlobalAvgPool
        → Dense head with dropout
        → Softmax output
    """
    inputs = tf.keras.Input(shape=input_shape, name="flow_features")

    # ── Block 1 ───────────────────────────────────────────────────────────
    x = layers.Conv1D(64, kernel_size=3, padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(64, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.2)(x)

    # ── Block 2 ───────────────────────────────────────────────────────────
    x = layers.Conv1D(128, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(128, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.2)(x)

    # ── Block 3 ───────────────────────────────────────────────────────────
    x = layers.Conv1D(256, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    # ── Head ──────────────────────────────────────────────────────────────
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(
        256,
        activation="relu",
        kernel_regularizer=regularizers.l2(1e-4),
    )(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(
        128,
        activation="relu",
        kernel_regularizer=regularizers.l2(1e-4),
    )(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = Model(inputs, outputs, name="CNN_NIDS")
    return model