"""Configuration constants for the brain tumor detector."""

from pathlib import Path

# --------------------------------------------------
# Base directory (Azure-safe)
# --------------------------------------------------
BASE_DIR = Path(__file__).parent

# --------------------------------------------------
# Model configuration (4-class classification)
# --------------------------------------------------
MODEL_PATH = BASE_DIR / "models" / "brain_tumor_multiclass.keras"
MODEL_PATH_LEGACY = BASE_DIR / "models" / "brain_tumor_ft.keras"

IMG_SIZE = (224, 224)

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
NUM_CLASSES = len(CLASS_NAMES)

# --------------------------------------------------
# Class display information
# --------------------------------------------------
CLASS_INFO = {
    "glioma": {
        "display": "Glioma",
        "type": "tumor",
        "description": "Brain tumor - Glioma",
    },
    "meningioma": {
        "display": "Meningioma",
        "type": "tumor",
        "description": "Brain tumor - Meningioma",
    },
    "notumor": {
        "display": "No Tumor",
        "type": "normal",
        "description": "No tumor detected",
    },
    "pituitary": {
        "display": "Pituitary",
        "type": "tumor",
        "description": "Brain tumor - Pituitary",
    },
}
