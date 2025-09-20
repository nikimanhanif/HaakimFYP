from pathlib import Path

import numpy as np
from PIL import Image

from stretch_detector.config import Config
from stretch_detector.data.video_dataset import (
    build_arrays_for_split,
    COMBINED_CLASS_KEYS,
    label_from_name,
    label_from_path,
)


def _make_frames(dir_path: Path, n: int, size=(50, 50)):
    dir_path.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        arr = (np.random.rand(size[0], size[1]) * 255).astype("uint8")
        Image.fromarray(arr).save(dir_path / f"frame_{i:03d}.png")


def test_build_arrays_empty_split(tmp_path: Path):
    cfg = Config(
        project_root=tmp_path,
        data_dir=tmp_path / "data",
        raw_videos_dir=tmp_path / "data" / "raw_videos",
        frames_dir=tmp_path / "data" / "frames",
        npz_dir=tmp_path / "data" / "npz",
        models_dir=tmp_path / "models",
        logs_dir=tmp_path / "logs",
    )
    cfg.ensure_dirs()
    X, y = build_arrays_for_split(cfg, "train")
    assert X.shape[0] == 0
    assert y.shape[0] == 0


def test_build_arrays_for_split_basic(tmp_path: Path):
    cfg = Config(
        data_dir=tmp_path,
        raw_videos_dir=tmp_path / "raw",
        frames_dir=tmp_path / "frames",
        npz_dir=tmp_path / "npz",
        models_dir=tmp_path / "models",
        logs_dir=tmp_path / "logs",
        seq_len=8,
        image_size=(50, 50),
        num_classes=10,
    )
    cfg.ensure_dirs()

    cls_a = COMBINED_CLASS_KEYS[0]
    cls_b = COMBINED_CLASS_KEYS[1]

    short_dir = cfg.frames_dir / "train" / f"{cls_a}_sample1"
    long_dir = cfg.frames_dir / "train" / f"{cls_b}_sample2"

    _make_frames(short_dir, cfg.seq_len - 3, size=cfg.image_size)
    _make_frames(long_dir, cfg.seq_len + 5, size=cfg.image_size)

    X_train, y_train = build_arrays_for_split(cfg, "train")
    assert X_train.shape[0] == 2
    assert X_train.shape[1:] == (cfg.seq_len, cfg.image_size[0], cfg.image_size[1], 1)
    assert set(y_train.tolist()) <= {COMBINED_CLASS_KEYS.index(cls_a), COMBINED_CLASS_KEYS.index(cls_b)}


def test_label_inference(tmp_path: Path):
    cls_key = COMBINED_CLASS_KEYS[3]
    p = tmp_path / cls_key
    p.mkdir()
    lbl_from_name = label_from_name(cls_key)
    lbl_from_path = label_from_path(p)
    assert lbl_from_name == lbl_from_path == COMBINED_CLASS_KEYS.index(cls_key)


def test_empty_split(tmp_path: Path):
    cfg = Config(
        data_dir=tmp_path,
        raw_videos_dir=tmp_path / "raw",
        frames_dir=tmp_path / "frames",
        npz_dir=tmp_path / "npz",
        models_dir=tmp_path / "models",
        logs_dir=tmp_path / "logs",
        seq_len=6,
        image_size=(32, 32),
        num_classes=10,
    )
    cfg.ensure_dirs()
    X_val, y_val = build_arrays_for_split(cfg, "val")
    assert X_val.shape[0] == 0
    assert y_val.shape[0] == 0
