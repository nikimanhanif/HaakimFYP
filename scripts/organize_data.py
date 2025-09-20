#!/usr/bin/env python3
from __future__ import annotations
import argparse
import shutil
from pathlib import Path

from stretch_detector.data.video_dataset import (
    COMBINED_CLASS_KEYS,
    label_from_name,
    label_from_path,
    class_key_from_label,
)

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv"}


def infer_class_key(p: Path) -> str | None:
    # Try full path first, then filename
    lbl = label_from_path(p)
    if lbl is None:
        lbl = label_from_name(p.name)
    if lbl is None:
        return None
    key = class_key_from_label(lbl)
    # Map legacy keys to combined naming if needed
    if key in ("st_", "no_st_"):
        return key  # legacy binary (kept as-is)
    return key


def unique_dest(dest: Path) -> Path:
    if not dest.exists():
        return dest
    stem, suf = dest.stem, dest.suffix
    i = 1
    while True:
        cand = dest.with_name(f"{stem}__{i}{suf}")
        if not cand.exists():
            return cand
        i += 1


def plan(src: Path, dst: Path, flat: bool) -> tuple[list[tuple[Path, Path]], list[Path]]:
    plans: list[tuple[Path, Path]] = []
    unknown: list[Path] = []
    for p in src.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in VIDEO_EXTS:
            continue
        key = infer_class_key(p)
        if key is None:
            unknown.append(p)
            continue
        if flat:
            out_dir = dst
            out_name = f"{key}__{p.name}"
        else:
            out_dir = dst / key
            out_name = p.name
        out_dir.mkdir(parents=True, exist_ok=True)
        dest = unique_dest(out_dir / out_name)
        # Skip if already at destination path
        if p.resolve() == dest.resolve():
            continue
        plans.append((p, dest))
    return plans, unknown


def main() -> None:
    ap = argparse.ArgumentParser(description="Organize raw videos into a consistent 10-class structure.")
    ap.add_argument("--src", type=Path, default=Path("data/raw"), help="Source root to scan for videos")
    ap.add_argument("--dst", type=Path, default=Path("data/organized"), help="Destination root to place organized videos")
    ap.add_argument("--flat", action="store_true", help="Flatten into a single folder with <class_key>__filename")
    ap.add_argument("--move", action="store_true", help="Move files instead of copying")
    ap.add_argument("--dry-run", action="store_true", help="Preview actions without making changes")
    args = ap.parse_args()

    plans, unknown = plan(args.src, args.dst, flat=args.flat)

    print(f"Detected {len(plans)} file(s) to {'move' if args.move else 'copy'} into {args.dst}")
    if unknown:
        print(f"Warning: {len(unknown)} file(s) had unknown class and will be skipped.")
        for p in unknown[:10]:
            print(f"  - Unknown: {p}")
        if len(unknown) > 10:
            print(f"  ... and {len(unknown)-10} more")

    if args.dry_run:
        for src, dst in plans[:20]:
            print(f"PLAN: {'MOVE' if args.move else 'COPY'} {src} -> {dst}")
        if len(plans) > 20:
            print(f"... and {len(plans)-20} more")
        return

    for src_p, dst_p in plans:
        dst_p.parent.mkdir(parents=True, exist_ok=True)
        if args.move:
            shutil.move(str(src_p), str(dst_p))
        else:
            shutil.copy2(src_p, dst_p)

    print("Done.")


if __name__ == "__main__":
    main()
