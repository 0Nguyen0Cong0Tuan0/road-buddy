from pathlib import Path


current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent

TRAIN_DIR = parent_dir / "data" / "train"
TRAIN_VIDEO = TRAIN_DIR / "videos"
TRAIN_JSON = TRAIN_DIR / "train.json"
