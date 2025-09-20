from __future__ import annotations
from typing import Tuple, Union

from tensorflow import keras
from tensorflow.keras import layers


def build_cnn3d(
    seq_len: Union[int, Tuple[int, int, int], Tuple[int, int, int, int]] = 10,
    img_size: tuple[int, int] = (250, 250),
    num_classes: int = 1,
) -> keras.Model:
    """Build a 3D CNN model.

    Accepts either an input_shape tuple as the first argument, e.g. (T, H, W, C) or (T, H, W),
    or the legacy (seq_len: int, img_size: (W, H)) arguments.
    """
    # Parse input shape
    if isinstance(seq_len, (tuple, list)):
        if len(seq_len) == 4:
            T, H, W, C = map(int, seq_len)
        elif len(seq_len) == 3:
            T, H, W = map(int, seq_len)
            C = 1
        else:
            raise ValueError("input shape must be (T,H,W[,C])")
    else:
        T = int(seq_len)
        W = int(img_size[0])
        H = int(img_size[1])
        C = 1

    inputs = keras.Input(shape=(T, H, W, C), name="video")
    x = layers.Conv3D(32, (3, 3, 3), padding="same", activation="relu")(inputs)
    x = layers.MaxPooling3D((2, 2, 2))(x)
    x = layers.Conv3D(64, (3, 3, 3), padding="same", activation="relu")(x)
    x = layers.MaxPooling3D((2, 2, 2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dense(32, activation="relu")(x)
    outputs = layers.Dense(num_classes, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs, name="cnn3d")
    return model
