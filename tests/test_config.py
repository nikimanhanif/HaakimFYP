import os
from pathlib import Path

from stretch_detector.config import DEFAULT_CONFIG, Config


def test_config_dirs(tmp_path: Path):
    cfg = Config(
        project_root=tmp_path,
        data_dir=tmp_path / "data",
        raw_videos_dir=tmp_path / "data" / "raw_videos",
        frames_dir=tmp_path / "data" / "frames",
        npz_dir=tmp_path / "data" / "npz",
        models_dir=tmp_path / "models",
        logs_dir=tmp_path / "logs",
    )
    cfg.ensure_dirs()
    assert cfg.data_dir.exists()
    assert (cfg.frames_dir / "train").exists()
    assert (cfg.frames_dir / "val").exists()
    assert (cfg.frames_dir / "test").exists()
