#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, List

from stretch_detector.config import DEFAULT_CONFIG
from stretch_detector.data.video_dataset import label_from_path, label_from_name

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}


def build_raw_stem_labels(raw_root: Path) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    for p in raw_root.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in VIDEO_EXTS:
            continue
        lbl = label_from_path(p)
        if lbl is None:
            lbl = label_from_name(p.name)
        if lbl is None:
            continue
        mapping[p.stem] = int(lbl)
    return mapping


essential_splits = ("train", "val", "test")


def write_labels_for_split(frames_split_dir: Path, stem_to_label: Dict[str, int]) -> int:
    frames_split_dir.mkdir(parents=True, exist_ok=True)
    # Identify sample directories (direct children with frames inside)
    sample_dirs: List[Path] = [d for d in frames_split_dir.iterdir() if d.is_dir()]
    out: Dict[str, int] = {}
    missing = 0
    for d in sample_dirs:
        lbl = stem_to_label.get(d.name)
        if lbl is None:
            # leave missing to be inferred later (may be 0)
            missing += 1
            continue
        out[d.name] = int(lbl)
    with open(frames_split_dir / "_labels.json", "w", encoding="utf-8") as f:
        json.dump(out, f)
    return missing


def main() -> None:
    ap = argparse.ArgumentParser(description="Write _labels.json for each split under data/frames using raw video names.")
    ap.add_argument("--raw", type=Path, default=None, help="Raw videos root (defaults to cfg.raw_videos_dir)")
    ap.add_argument("--frames", type=Path, default=None, help="Frames root (defaults to cfg.frames_dir)")
    args = ap.parse_args()

    cfg = DEFAULT_CONFIG
    raw_root = args.raw or cfg.raw_videos_dir
    frames_root = args.frames or cfg.frames_dir

    stem_to_label = build_raw_stem_labels(raw_root)
    print(f"Raw stems with labels: {len(stem_to_label)}")

    for split in essential_splits:
        split_dir = frames_root / split
        if not split_dir.exists():
            print(f"Skip {split}: {split_dir} not found")
            continue
        missing = write_labels_for_split(split_dir, stem_to_label)
        total = len([d for d in split_dir.iterdir() if d.is_dir()])
        print(f"Wrote labels for {split}: {total - missing}/{total} matched; missing={missing}")


if __name__ == "__main__":
    main()
