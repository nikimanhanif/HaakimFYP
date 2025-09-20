#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path

from stretch_detector.config import DEFAULT_CONFIG
from stretch_detector.eval.evaluate import evaluate_model


def main():
    parser = argparse.ArgumentParser(description="Evaluate a saved model on the test split.")
    parser.add_argument("--model_path", type=str, required=True)
    args = parser.parse_args()

    cfg = DEFAULT_CONFIG
    metrics = evaluate_model(cfg, Path(args.model_path), split="test")
    print(metrics)


if __name__ == "__main__":
    main()
