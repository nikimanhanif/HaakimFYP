#!/usr/bin/env python3
from __future__ import annotations
import sys

# Smoke import test to validate package structure
try:
    import stretch_detector
    from stretch_detector.config import DEFAULT_CONFIG
    from stretch_detector.data.video_dataset import VideoDataset
    from stretch_detector.models.cnn_lstm import build_cnn_lstm
    from stretch_detector.models.cnn3d import build_cnn3d
    print("OK: imports passed")
except Exception as e:
    print(f"Import error: {e}")
    sys.exit(1)
