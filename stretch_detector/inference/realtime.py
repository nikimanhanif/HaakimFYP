from __future__ import annotations
import math
from pathlib import Path
from typing import List, Tuple, Optional, Union

import cv2
import numpy as np
from moviepy.video.io.VideoFileClip import VideoFileClip
from tensorflow import keras

from ..config import Config


def segment_video(input_video: Path, out_dir: Path, segment_seconds: int = 5, max_segments: int | None = None) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    with VideoFileClip(str(input_video)) as clip:
        duration = int(math.floor(clip.duration))
        n_segments = duration // segment_seconds
        if max_segments is not None:
            n_segments = min(n_segments, max_segments)
        seg_paths: List[Path] = []
        for i in range(n_segments):
            start = i * segment_seconds
            end = start + segment_seconds
            seg_path = out_dir / f"seg_{i}.mp4"
            sub = clip.subclip(start, end)
            sub.write_videofile(str(seg_path), audio_codec="aac", verbose=False, logger=None)
            seg_paths.append(seg_path)
    return seg_paths


def _resolve_cfg_values(
    cfg: Config | None,
    fallback_seq_len: int = 10,
    fallback_size: Tuple[int, int] = (250, 250),
) -> Tuple[int, Tuple[int, int]]:
    """Best-effort extraction of seq_len and image size from a Config-like object."""
    seq_len = fallback_seq_len
    size = fallback_size

    if cfg is not None:
        # Resolve seq_len
        for attr in ("seq_len", "sequence_length", "frames_per_clip", "num_frames"):
            if hasattr(cfg, attr):
                try:
                    seq_len = int(getattr(cfg, attr))
                    break
                except Exception:
                    pass
        # Resolve size (prefer a tuple/list like (W, H) or (H, W))
        for attr in ("image_size", "frame_size", "size", "img_size", "input_size"):
            if hasattr(cfg, attr):
                val = getattr(cfg, attr)
                if isinstance(val, (tuple, list)) and len(val) >= 2:
                    try:
                        size = (int(val[0]), int(val[1]))
                        break
                    except Exception:
                        pass
        # Some configs may have separate width/height
        if size == fallback_size:
            w = getattr(cfg, "width", None) or getattr(cfg, "img_width", None)
            h = getattr(cfg, "height", None) or getattr(cfg, "img_height", None)
            if w and h:
                try:
                    size = (int(w), int(h))
                except Exception:
                    pass

    return seq_len, size


def _frames_from_video(
    video_path: str,
    cfg_or_seq: Union[Config, int, None] = None,
    size: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """
    Load frames from a video, convert to grayscale, resize, and ensure exactly seq_len frames.
    Accepts either:
      - cfg_or_seq as Config (seq_len and size inferred), or
      - cfg_or_seq as int (seq_len) plus optional size tuple.
    Returns array of shape (seq_len, H, W, 1), dtype float32 in [0, 1].
    """
    # Derive seq_len and size
    if isinstance(cfg_or_seq, Config):
        seq_len, inferred_size = _resolve_cfg_values(cfg_or_seq)
        target_size = size or inferred_size
    else:
        seq_len = int(cfg_or_seq) if isinstance(cfg_or_seq, int) else 10
        target_size = size or (250, 250)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    frames: list[np.ndarray] = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # OpenCV expects (width, height)
            resized = cv2.resize(gray, (int(target_size[0]), int(target_size[1])), interpolation=cv2.INTER_AREA)
            frames.append(np.expand_dims(resized, axis=-1))  # (H, W, 1)
    finally:
        cap.release()

    if len(frames) == 0:
        frames = [np.zeros((int(target_size[1]), int(target_size[0]), 1), dtype=np.uint8)]
        # Note: above uses (H, W, 1); for square sizes this is equivalent.

    n = len(frames)
    if n < seq_len:
        padded = frames.copy()
        i = 0
        while len(padded) < seq_len:
            padded.append(frames[i % n])
            i += 1
        frames = padded
    elif n > seq_len:
        idx = np.linspace(0, n - 1, num=seq_len).astype(int).tolist()
        frames = [frames[i] for i in idx]

    arr = np.stack(frames, axis=0)  # (seq_len, H, W, 1)
    arr = arr.astype(np.float32) / 255.0
    return arr


def predict_segments(cfg: Config, model_path: Path, input_video: Path, work_dir: Path, segment_seconds: int = 5) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    model = keras.models.load_model(str(model_path))
    seg_dir = work_dir / (input_video.stem + "_segments")
    seg_paths = segment_video(input_video, seg_dir, segment_seconds=segment_seconds)

    preds = []
    windows: List[Tuple[int, int]] = []
    for i, seg in enumerate(seg_paths):
        X = _frames_from_video(seg, cfg)[None, ...]  # (1, T, H, W, 1)
        p = model.predict(X, verbose=0)[0, 0]
        preds.append(p)
        windows.append((i * segment_seconds, (i + 1) * segment_seconds))
    return np.array(preds, dtype=float), windows
