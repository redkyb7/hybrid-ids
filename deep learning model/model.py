# model.py

import tensorflow as tf

from tensorflow.keras import (
    Model,
    layers,
    regularizers,
)


def build_cnn(
    input_shape: tuple,
    num_classes: int,
) -> Model:
    """
    Efficient 1D CNN for network-flow
    intrusion detection.

    Input:
        (52 features, 1 channel)

    Architecture:
        Conv1D 32
        ↓
        BatchNorm
        ↓
        MaxPool
        ↓
        Conv1D 64
        ↓
        BatchNorm
        ↓
        MaxPool
        ↓
        Conv1D 128
        ↓
        BatchNorm
        ↓
        GlobalAveragePooling
        ↓
        Dense 128
        ↓
        Dense 64
        ↓
        Softmax
    """

    # ========================================================
    # INPUT
    # ========================================================

    inputs = tf.keras.Input(
        shape=input_shape,
        name="flow_features",
    )


    # ========================================================
    # CONVOLUTION BLOCK 1
    # ========================================================

    x = layers.Conv1D(
        filters=32,
        kernel_size=3,
        padding="same",
        activation="relu",
    )(inputs)

    x = layers.BatchNormalization()(x)

    x = layers.MaxPooling1D(
        pool_size=2,
    )(x)


    # ========================================================
    # CONVOLUTION BLOCK 2
    # ========================================================

    x = layers.Conv1D(
        filters=64,
        kernel_size=3,
        padding="same",
        activation="relu",
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.MaxPooling1D(
        pool_size=2,
    )(x)


    # ========================================================
    # CONVOLUTION BLOCK 3
    # ========================================================

    x = layers.Conv1D(
        filters=128,
        kernel_size=3,
        padding="same",
        activation="relu",
    )(x)

    x = layers.BatchNormalization()(x)


    # ========================================================
    # GLOBAL POOLING
    # ========================================================

    x = layers.GlobalAveragePooling1D()(x)


    # ========================================================
    # CLASSIFICATION HEAD
    # ========================================================

    x = layers.Dense(
        128,
        activation="relu",
        kernel_regularizer=regularizers.l2(
            1e-4
        ),
    )(x)

    x = layers.Dropout(
        0.30
    )(x)

    x = layers.Dense(
        64,
        activation="relu",
    )(x)

    x = layers.Dropout(
        0.20
    )(x)


    # ========================================================
    # OUTPUT
    # ========================================================

    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        name="predictions",
    )(x)


    model = Model(
        inputs=inputs,
        outputs=outputs,
        name="CNN_NIDS",
    )

    return model