from __future__ import annotations
from pathlib import Path

import numpy as np
from tensorflow import keras

from ..config import Config
from ..data.video_dataset import build_arrays_for_split


def evaluate_model(cfg: Config, model_path: Path, split: str = "test") -> dict:
    model = keras.models.load_model(str(model_path))
    X, y = build_arrays_for_split(cfg, split)
    if X.shape[0] == 0:
        raise RuntimeError(f"No data found in split '{split}'. Ensure frames are prepared and named with st_/no_st_.")
    loss, acc = model.evaluate(X, y, verbose=0)
    return {"loss": float(loss), "accuracy": float(acc), "n_samples": int(X.shape[0])}
