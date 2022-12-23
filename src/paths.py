from pathlib import Path


SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent
MODELS_DIR = ROOT_DIR / "models"
OUTPUTS_DIR = ROOT_DIR / "outputs"


ALL_DIRS = [
    SRC_DIR,
    ROOT_DIR,
    MODELS_DIR
]

for DIR in ALL_DIRS:
    DIR.mkdir(exist_ok=True, parents=True)