from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


@dataclass
class Config:
    # Paths
    project_root: Path = Path(__file__).resolve().parents[1]
    data_dir: Path = project_root / "data"
    raw_videos_dir: Path = project_root / "data" / "organized"
    frames_dir: Path = project_root / "data" / "frames"
    npz_dir: Path = project_root / "data" / "npz"
    models_dir: Path = project_root / "models"
    logs_dir: Path = project_root / "logs"

    # Data
    seq_len: int = 10
    image_size: Tuple[int, int] = (250, 250)  # (H, W)
    channels: int = 1  # grayscale
    fps_sample: int = 1  # approx frames per second to sample

    # Labels
    num_classes: int = 1  # 1=binary, >1=multiclass (e.g., 10 for combined classes)

    # Splits
    test_ratio: float = 0.26
    val_ratio: float = 0.2
    random_seed: int = 14

    # Training
    batch_size: int = 10
    epochs: int = 20
    learning_rate: float = 0.01

    def ensure_dirs(self):
        from pathlib import Path as _P
        # Create top-level dirs
        for p in [self.data_dir, self.raw_videos_dir, self.frames_dir, self.npz_dir, self.models_dir, self.logs_dir]:
            _P(p).mkdir(parents=True, exist_ok=True)
        # Create split subdirs
        for split in ["train", "val", "test"]:
            (self.frames_dir / split).mkdir(parents=True, exist_ok=True)
            (self.npz_dir / split).mkdir(parents=True, exist_ok=True)


DEFAULT_CONFIG = Config(
    num_classes=10,  # combined 10-class setup
)
