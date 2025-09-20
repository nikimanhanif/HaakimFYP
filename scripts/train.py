#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import math  # NEW

import numpy as np
from tensorflow import keras

from stretch_detector.config import DEFAULT_CONFIG, Config
from stretch_detector.data.video_dataset import build_arrays_for_split
from stretch_detector.models.cnn_lstm import build_cnn_lstm
from stretch_detector.models.cnn3d import build_cnn3d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["cnn_lstm", "cnn3d"], default="cnn_lstm")
    ap.add_argument("--num-classes", type=int, default=None, help="Override cfg.num_classes")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    args = ap.parse_args()

    cfg: Config = DEFAULT_CONFIG
    if args.num_classes is not None:
        cfg.num_classes = int(args.num_classes)
    if args.epochs is not None:
        cfg.epochs = int(args.epochs)
    if args.batch_size is not None:
        cfg.batch_size = int(args.batch_size)

    cfg.ensure_dirs()

    # Load arrays
    X_train, y_train = build_arrays_for_split(cfg, "train")
    X_val, y_val = build_arrays_for_split(cfg, "val")

    n_train = int(getattr(X_train, "shape", (0,))[0] or 0)
    n_val = int(getattr(X_val, "shape", (0,))[0] or 0)

    if n_train <= 0:
        raise RuntimeError(
            f"No training samples found. Check prepare_data output and paths.\n"
            f"frames_dir={cfg.frames_dir}, npz_dir={cfg.npz_dir}, num_classes={cfg.num_classes}"
        )

    # Clamp batch size to ensure at least 1 step
    batch_size = max(1, min(cfg.batch_size, n_train))

    # Build model
    input_shape = (cfg.seq_len, cfg.image_size[0], cfg.image_size[1], cfg.channels)  # (T,H,W,C)
    if args.model == "cnn_lstm":
        model = build_cnn_lstm(input_shape, num_classes=cfg.num_classes)
    else:
        model = build_cnn3d(input_shape, num_classes=cfg.num_classes)

    # Compile with correct loss
    if cfg.num_classes == 1:
        loss = "binary_crossentropy"; metrics = ["accuracy"]
    else:
        loss = "sparse_categorical_crossentropy"; metrics = ["accuracy"]

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=cfg.learning_rate),
                  loss=loss, metrics=metrics)

    # Callbacks (switch monitor if no val set)
    ckpt_path = cfg.models_dir / f"best_{args.model}_{cfg.num_classes}cls.keras"
    monitor = "val_loss" if n_val > 0 else "loss"
    callbacks = [
        keras.callbacks.ModelCheckpoint(filepath=str(ckpt_path), save_best_only=True, monitor=monitor),
        keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor=monitor),
    ]

    if n_val > 0:
        val_data = (X_val, y_val)
    else:
        val_data = None

    model.fit(
        X_train, y_train,
        validation_data=val_data,
        epochs=cfg.epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
        shuffle=True,
    )

    # Save final model too
    final_path = cfg.models_dir / f"final_{args.model}_{cfg.num_classes}cls.keras"
    model.save(final_path)
    print(f"Saved best to: {ckpt_path}")
    print(f"Saved final to: {final_path}")


if __name__ == "__main__":
    main()
