from __future__ import annotations
import os
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable
from PIL import Image
import numpy as np

from ..config import Config
from .frame_extractor import FrameExtractor

# Treat these as image frame extensions (produced by prepare step)
_IMG_EXTS = {".png", ".jpg", ".jpeg"}

@dataclass
class SplitPaths:
    train: List[Path]
    val: List[Path]
    test: List[Path]

# Combined 10-class keys (exercise + form). Lowercase, used as filename prefixes.
COMBINED_CLASS_KEYS: List[str] = [
    "barbell_bicep_curl_proper",
    "barbell_bicep_curl_improper",
    "bench_press_proper",
    "bench_press_improper",
    "deadlift_proper",
    "deadlift_improper",
    "plank_proper",
    "plank_improper",
    "squat_proper",
    "squat_improper",
]

# Simple synonym maps for robustness
_EXERCISE_ALIASES = {
    "barbell_bicep_curl": {"barbell_bicep_curl", "bicep_curl", "curl"},
    "bench_press": {"bench_press", "bench"},
    "deadlift": {"deadlift", "deadlifts"},
    "plank": {"plank"},
    "squat": {"squat", "squats"},
}
_FORM_ALIASES = {
    "proper": {"proper", "correct", "good"},
    "improper": {"improper", "incorrect", "bad", "wrong"},
}

def _match_token(tokens: List[str], choices: dict) -> Optional[str]:
    toks = [t.lower() for t in tokens]
    for key, aliases in choices.items():
        if any(a in toks for a in aliases):
            return key
    return None

def label_from_name(name: str) -> Optional[int]:
    """Infer class label from a filename or directory name using aliases."""
    base = name.lower()
    tokens = [t for t in base.replace("-", "_").replace(" ", "_").split("_") if t]
    ex = _match_token(tokens, _EXERCISE_ALIASES)
    form = _match_token(tokens, _FORM_ALIASES)
    if ex and form:
        key = f"{ex}_{form}"
        # alias normalization: 'deadlift' key in COMBINED uses singular
        if key not in COMBINED_CLASS_KEYS and key.startswith("deadlift_"):
            key = key  # already normalized
        if key in COMBINED_CLASS_KEYS:
            return COMBINED_CLASS_KEYS.index(key)
    # Try exact 10-class keys if embedded
    for i, k in enumerate(COMBINED_CLASS_KEYS):
        if k in base:
            return i
    return None

def label_from_path(path: Path) -> Optional[int]:
    """Walk up the path to find a matching label from any folder/file name."""
    for p in [path] + list(path.parents):
        lbl = label_from_name(p.name)
        if lbl is not None:
            return lbl
    return None

def class_key_from_label(label: int) -> str:
    return COMBINED_CLASS_KEYS[int(label)]

def _dir_has_images(p: Path) -> bool:
    if not p.is_dir():
        return False
    for f in p.iterdir():
        if f.is_file() and f.suffix.lower() in _IMG_EXTS:
            return True
    return False

def _iter_sample_dirs(frames_split_dir: Path) -> List[Path]:
    """
    Find sample directories in frames_split_dir by locating directories
    that contain frame images. Works for layouts like:
      frames/train/<class_key>/<video_stem>/*.png
      frames/train/<video_stem>/*.png
    """
    if not frames_split_dir.exists():
        return []
    sample_dirs: List[Path] = []
    for p in frames_split_dir.rglob("*"):
        if _dir_has_images(p):
            sample_dirs.append(p)
    return sample_dirs

def _load_frames_dir(dir_path: Path, cfg: Config) -> Optional[np.ndarray]:
    """Load frames from a directory into shape (T,H,W,1) float32 [0,1], pad/trim to cfg.seq_len."""
    files = sorted([f for f in dir_path.iterdir() if f.is_file() and f.suffix.lower() in _IMG_EXTS])
    if not files:
        return None
    frames: List[np.ndarray] = []
    H, W = int(cfg.image_size[0]), int(cfg.image_size[1])  # (H, W)
    for f in files:
        with Image.open(f) as im:
            im = im.convert("L")
            im = im.resize((W, H))
            arr = np.array(im, dtype=np.uint8)
            frames.append(arr[..., None])  # (H,W,1)
    # Normalize length
    T = cfg.seq_len
    n = len(frames)
    if n < T:
        padded = frames.copy()
        i = 0
        while len(padded) < T:
            padded.append(frames[i % n])
            i += 1
        frames = padded
    elif n > T:
        idx = np.linspace(0, n - 1, num=T).astype(int).tolist()
        frames = [frames[i] for i in idx]
    x = np.stack(frames, axis=0).astype(np.float32) / 255.0  # (T,H,W,1)
    return x


def _labels_file(split_dir: Path) -> Path:
    return split_dir / "_labels.json"


def _write_labels_json(cfg: Config, split: str, videos: List[Path]) -> None:
    """Persist mapping from video stem -> class id for a split.

    We write this before extracting frames so later loading can map frame folders
    (named by video stem) back to class labels even if names have no class tokens.
    """
    mapping: Dict[str, int] = {}
    for vp in videos:
        lbl = label_from_path(vp)
        if lbl is None:
            lbl = label_from_name(vp.name)
        if lbl is None:
            continue
        mapping[vp.stem] = int(lbl)
    split_dir = cfg.frames_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    with open(_labels_file(split_dir), "w", encoding="utf-8") as f:
        json.dump(mapping, f)

def build_arrays_for_split(cfg: Config, split: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build X, y arrays for a split from frames_dir, caching to npz_dir.
    Returns:
      X: (N, T, H, W, 1) float32
      y: (N,) int64 class ids in [0..num_classes-1]
    """
    split = split.lower()
    assert split in ("train", "val", "test")
    cache = cfg.npz_dir / split / "data.npz"
    if cache.exists():
        data = np.load(cache)
        X, y = data["X"], data["y"]
        return X, y

    frames_split_dir = cfg.frames_dir / split
    # Load labels mapping if available
    labels_map: Dict[str, int] = {}
    lf = _labels_file(frames_split_dir)
    if lf.exists():
        try:
            with open(lf, "r", encoding="utf-8") as f:
                raw = json.load(f)
                labels_map = {str(k): int(v) for k, v in raw.items()}
        except Exception:
            labels_map = {}
    sample_dirs = _iter_sample_dirs(frames_split_dir)
    X_list: List[np.ndarray] = []
    y_list: List[int] = []
    for sample_dir in sample_dirs:
        # Prefer mapping by directory name (video stem)
        stem = sample_dir.name
        lbl = labels_map.get(stem)
        if lbl is None:
            lbl = label_from_path(sample_dir)
        if lbl is None:
            # Try using immediate parent or dir name
            lbl = label_from_name(sample_dir.name)
        if lbl is None:
            continue
        # Enforce num_classes bound (skip unknown classes)
        if lbl < 0 or lbl >= max(cfg.num_classes, len(COMBINED_CLASS_KEYS)):
            continue
        x = _load_frames_dir(sample_dir, cfg)
        if x is None:
            continue
        X_list.append(x)
        y_list.append(int(lbl))

    if not X_list:
        # Empty split
        return np.empty((0, cfg.seq_len, cfg.image_size[0], cfg.image_size[1], cfg.channels), dtype=np.float32), np.empty((0,), dtype=np.int64)

    X = np.stack(X_list, axis=0).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)

    # Cache
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, X=X, y=y)
    return X, y


# ---- Added convenience prepare step to restore prior CLI workflow ----
_VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".mpeg", ".mpg"}


def _gather_videos(split_dir: Path) -> List[Path]:
    if not split_dir.exists():
        return []
    vids: List[Path] = []
    for p in split_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in _VIDEO_EXTS:
            vids.append(p)
    return vids


def prepare_frames(cfg: Config) -> SplitPaths:
    """Scan raw_videos_dir for train/val/test splits, infer labels, write label maps, and extract frames.

    Layout expected (flexible nesting allowed):
        raw/
          train/ <videos or folders>
          val/   <videos or folders>
          test/  <videos or folders>

    For each split we:
      1. Collect all video files by extension.
      2. Write a _labels.json mapping (video stem -> class id) for future frame directory resolution.
      3. Extract ~1 fps grayscale frames into frames/<split>/<video_stem>/frame#.jpg

    Returns a SplitPaths of the original video file lists for reference.
    """
    fx = FrameExtractor(cfg)
    raw_root = cfg.raw_videos_dir
    train_videos = _gather_videos(raw_root / "train")
    val_videos = _gather_videos(raw_root / "val")
    test_videos = _gather_videos(raw_root / "test")

    # Persist label maps BEFORE extraction so frame dirs (video stems) are mapped.
    if train_videos:
        _write_labels_json(cfg, "train", train_videos)
    if val_videos:
        _write_labels_json(cfg, "val", val_videos)
    if test_videos:
        _write_labels_json(cfg, "test", test_videos)

    if train_videos:
        fx.extract_split(train_videos, "train")
    if val_videos:
        fx.extract_split(val_videos, "val")
    if test_videos:
        fx.extract_split(test_videos, "test")

    return SplitPaths(train=train_videos, val=val_videos, test=test_videos)

