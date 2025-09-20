from __future__ import annotations
import math
import os
from pathlib import Path
from typing import Iterable

import cv2

from ..config import Config


class FrameExtractor:
    """Extract grayscale frames for each video into frames_dir/split/<video_name>/frame#.jpg
    Samples ~1 frame per second by using CAP_PROP_FPS and modulo.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.cfg.ensure_dirs()

    def extract_split(self, videos: Iterable[Path], split: str) -> None:
        out_root = self.cfg.frames_dir / split
        out_root.mkdir(parents=True, exist_ok=True)
        for vid in videos:
            self._extract_video(vid, out_root)

    def _extract_video(self, video_path: Path, out_root: Path) -> None:
        name = video_path.stem
        out_dir = out_root / name
        out_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video_path))
        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            fps_int = max(1, int(math.floor(fps)))
            frame_id = 0
            saved = 0
            while cap.isOpened():
                ok, frame = cap.read()
                if not ok:
                    break
                if frame_id % fps_int == 0:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    out_file = out_dir / f"frame{saved}.jpg"
                    cv2.imwrite(str(out_file), gray)
                    saved += 1
                frame_id += 1
        finally:
            cap.release()
