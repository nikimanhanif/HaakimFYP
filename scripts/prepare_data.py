#!/usr/bin/env python3
from __future__ import annotations
import argparse

from stretch_detector.config import DEFAULT_CONFIG
from stretch_detector.data.video_dataset import prepare_frames


def main():
    parser = argparse.ArgumentParser(description="Prepare dataset: scan raw videos and extract frames for splits.")
    args = parser.parse_args()
    cfg = DEFAULT_CONFIG
    cfg.ensure_dirs()
    splits = prepare_frames(cfg)
    print(f"Prepared frames: train={len(splits.train)}, val={len(splits.val)}, test={len(splits.test)}")


if __name__ == "__main__":
    main()
