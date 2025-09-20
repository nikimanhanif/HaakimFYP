from pathlib import Path

import numpy as np
import pytest

from stretch_detector.config import Config
from stretch_detector.inference.realtime import _frames_from_video

# Skip OpenCV-dependent test if cv2 not installed
try:
    import cv2  # noqa: F401
    CV2_AVAILABLE = True
except Exception:
    CV2_AVAILABLE = False


@pytest.mark.skipif(not CV2_AVAILABLE, reason="OpenCV not installed")
def test_frames_padding(tmp_path: Path):
    # Create a 1-frame dummy image and write a very short video using cv2 VideoWriter
    video_path = tmp_path / "short.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(video_path), fourcc, 5.0, (50, 50))
    frame = np.zeros((50, 50, 3), dtype=np.uint8)
    out.write(frame)
    out.release()

    cfg = Config(
        project_root=tmp_path,
        data_dir=tmp_path / "data",
        raw_videos_dir=tmp_path / "data" / "raw_videos",
        frames_dir=tmp_path / "data" / "frames",
        npz_dir=tmp_path / "data" / "npz",
        models_dir=tmp_path / "models",
        logs_dir=tmp_path / "logs",
        seq_len=10,
        image_size=(50, 50),
    )
    arr = _frames_from_video(video_path, cfg)
    assert arr.shape == (10, 50, 50, 1)
