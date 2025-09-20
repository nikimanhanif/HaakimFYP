#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path

from stretch_detector.config import DEFAULT_CONFIG
from stretch_detector.inference.realtime import predict_segments


def main():
    parser = argparse.ArgumentParser(description="Segment-wise predictions on a long video.")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--input_video", type=str, required=True)
    parser.add_argument("--work_dir", type=str, default="./predictions")
    parser.add_argument("--segment_seconds", type=int, default=5)
    args = parser.parse_args()

    cfg = DEFAULT_CONFIG
    work_dir = Path(args.work_dir)
    preds, windows = predict_segments(cfg, Path(args.model_path), Path(args.input_video), work_dir, args.segment_seconds)
    print({"windows": windows, "predictions": preds.tolist()})


if __name__ == "__main__":
    main()
