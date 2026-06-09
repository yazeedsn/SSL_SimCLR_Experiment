from pathlib import Path
import torch 

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "outputs"
PLOTS_DIR = OUTPUT_DIR / "plots"

DATASET_PATH = DATA_DIR / "STL10"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"