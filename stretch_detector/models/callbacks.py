from __future__ import annotations
from pathlib import Path
from typing import List

from tensorflow import keras


def default_callbacks(save_path: Path) -> List[keras.callbacks.Callback]:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    return [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(filepath=str(save_path), monitor="val_loss", save_best_only=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=2)
    ]
