from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "outputs"
PLOTS_DIR = OUTPUT_DIR / "plots"
CHECKPOINTS_DIR = OUTPUT_DIR / "checkpoints"

DATASET_PATH = DATA_DIR / "STL10"

