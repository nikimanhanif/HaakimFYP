from __future__ import annotations
from pathlib import Path

import numpy as np
from tensorflow import keras

from ..config import Config
from ..data.video_dataset import prepare_frames, build_arrays_for_split
from ..models.cnn3d import build_cnn3d
from ..models.callbacks import default_callbacks


def train(cfg: Config, num_classes: int = 1) -> Path:
    cfg.ensure_dirs()
    prepare_frames(cfg)

    X_train, y_train = build_arrays_for_split(cfg, "train")
    X_val, y_val = build_arrays_for_split(cfg, "val")

    model = build_cnn3d((cfg.seq_len, cfg.image_size[0], cfg.image_size[1], cfg.channels), num_classes=num_classes)
    opt = keras.optimizers.Adagrad(learning_rate=0.01)
    if num_classes == 1:
        loss = "binary_crossentropy"
        metrics = ["accuracy"]
    else:
        loss = "sparse_categorical_crossentropy"
        metrics = ["accuracy"]
    model.compile(optimizer=opt, loss=loss, metrics=metrics)

    ckpt_name = "cnn3d_best" if num_classes == 1 else f"cnn3d_best_{num_classes}c"
    ckpt_path = cfg.models_dir / f"{ckpt_name}.keras"
    cb = default_callbacks(ckpt_path)
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=10,
        epochs=15,
        callbacks=cb,
        verbose=2,
    )
    return ckpt_path
