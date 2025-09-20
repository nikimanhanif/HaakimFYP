# Stretch Detector: Video Classification (Stretching vs No-Stretching)

This codebase converts `DL_Stretch_updated.ipynb` into a modular Python project. It classifies short video clips as Stretching vs No-Stretching using:

- CNN+LSTM (TimeDistributed Conv2D + LSTM)
- 3D CNN (Conv3D)

Includes data prep (frame extraction), training, evaluation, and segment-wise real-time inference.

## Structure

- `stretch_detector/`
  - `config.py` — central configuration (paths, hyperparameters)
  - `data/`
    - `video_dataset.py` — scan videos, split, build tensors
    - `frame_extractor.py` — extract grayscale frames per video
  - `models/`
    - `cnn_lstm.py` — CNN+LSTM builder
    - `cnn3d.py` — 3D CNN builder
    - `callbacks.py` — Keras callbacks factory
  - `training/`
    - `train_cnn_lstm.py` — training entry for CNN+LSTM
    - `train_cnn3d.py` — training entry for 3D CNN
  - `eval/`
    - `evaluate.py` — test-set evaluation
  - `inference/`
    - `realtime.py` — segment-wise predictions for long videos
  - `utils/`
    - `plotting.py` — training curves (optional)
- `scripts/`
  - `prepare_data.py` — scan, split, and extract frames
  - `train.py` — train a selected model type
  - `evaluate.py` — evaluate a saved checkpoint
  - `predict_realtime.py` — run segment-wise predictions on a long video

## Quickstart

1) Put raw videos in `data/raw_videos/` with names like `st_*.mp4` or `no_st_*.mp4`.
2) Prepare frames and splits:
   - scripts/prepare_data.py --help
3) Train a model:
   - scripts/train.py --model cnn_lstm  (or cnn3d)
4) Evaluate a checkpoint:
   - scripts/evaluate.py --model_path models/cnn_lstm_best.keras
5) Segment-wise predictions for a long video:
   - scripts/predict_realtime.py --model_path models/cnn_lstm_best.keras --input_video your_long_video.mp4

Defaults use 10 frames (1 fps sampling) at 250x250 grayscale.

Note: On Apple Silicon, you may need `tensorflow-macos` and `tensorflow-metal`. If installing `tensorflow` fails, install those variants.
