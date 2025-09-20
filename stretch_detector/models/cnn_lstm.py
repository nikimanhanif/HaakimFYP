from __future__ import annotations
from typing import Tuple, Union

from tensorflow import keras
from tensorflow.keras import layers


def build_cnn_lstm(
    seq_len: Union[int, Tuple[int, int, int], Tuple[int, int, int, int]] = (10, 128, 128, 1),
    num_classes: int = 10,
) -> keras.Model:
    """Build a CNN+LSTM model.

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
    x = inputs
    for f in (24, 48, 96):
        x = layers.TimeDistributed(layers.Conv2D(f, 3, padding="same", activation="relu"))(x)
        x = layers.TimeDistributed(layers.Conv2D(f, 3, padding="same", activation="relu"))(x)
        x = layers.TimeDistributed(layers.MaxPooling2D(2))(x)
    x = layers.TimeDistributed(layers.GlobalAveragePooling2D())(x)  # (T,F)
    x = layers.LSTM(32, dropout=0.2, recurrent_dropout=0.1)(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x) if num_classes > 1 else layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs, name="cnn_lstm_medium")
    return model
