import tensorflow as tf

from tensorflow.keras import (
    Model,
    layers,
    regularizers,
)

if __package__:
    from . import config
else:
    import config


def residual_dense_block(
    x,
    units: int,
    dropout_rate: float,
    name: str,
):
    """
    Dense residual block suitable for tabular flow features.
    """

    shortcut = x

    x = layers.Dense(
        units,
        kernel_initializer="he_normal",
        kernel_regularizer=regularizers.l2(
            config.L2_REGULARIZATION
        ),
        name=f"{name}_dense_1",
    )(x)

    x = layers.BatchNormalization(
        name=f"{name}_batch_norm_1",
    )(x)

    x = layers.Activation(
        "relu",
        name=f"{name}_relu_1",
    )(x)

    x = layers.Dropout(
        dropout_rate,
        name=f"{name}_dropout_1",
    )(x)

    x = layers.Dense(
        units,
        kernel_regularizer=regularizers.l2(
            config.L2_REGULARIZATION
        ),
        name=f"{name}_dense_2",
    )(x)

    x = layers.BatchNormalization(
        name=f"{name}_batch_norm_2",
    )(x)

    if shortcut.shape[-1] != units:
        shortcut = layers.Dense(
            units,
            use_bias=False,
            name=f"{name}_projection",
        )(shortcut)

    x = layers.Add(
        name=f"{name}_add",
    )(
        [
            x,
            shortcut,
        ]
    )

    x = layers.Activation(
        "relu",
        name=f"{name}_relu_2",
    )(x)

    return x


def build_nids_model(
    input_shape: tuple,
    num_classes: int,
) -> Model:
    """
    Residual MLP for tabular network-flow intrusion detection.

    Input:
        (number_of_features, 1)

    The feature axis is flattened because flow statistics are tabular
    variables and do not form a natural time-series sequence.
    """

    inputs = tf.keras.Input(
        shape=input_shape,
        name="flow_features",
    )

    x = layers.Flatten(
        name="flatten_features",
    )(inputs)

    x = layers.Dense(
        256,
        kernel_initializer="he_normal",
        kernel_regularizer=regularizers.l2(
            config.L2_REGULARIZATION
        ),
        name="input_dense",
    )(x)

    x = layers.BatchNormalization(
        name="input_batch_norm",
    )(x)

    x = layers.Activation(
        "relu",
        name="input_relu",
    )(x)

    x = layers.Dropout(
        config.DROPOUT_INPUT,
        name="input_dropout",
    )(x)

    x = residual_dense_block(
        x,
        units=256,
        dropout_rate=config.DROPOUT_BLOCK_1,
        name="residual_block_1",
    )

    x = residual_dense_block(
        x,
        units=128,
        dropout_rate=config.DROPOUT_BLOCK_2,
        name="residual_block_2",
    )

    x = layers.Dense(
        64,
        activation="relu",
        kernel_regularizer=regularizers.l2(
            config.L2_REGULARIZATION
        ),
        name="classification_dense",
    )(x)

    x = layers.Dropout(
        0.15,
        name="classification_dropout",
    )(x)

    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        name="predictions",
    )(x)

    return Model(
        inputs=inputs,
        outputs=outputs,
        name="Tabular_Residual_NIDS",
    )


# Compatibility alias if older code imports build_cnn.
def build_cnn(
    input_shape: tuple,
    num_classes: int,
) -> Model:
    return build_nids_model(
        input_shape=input_shape,
        num_classes=num_classes,
    )
