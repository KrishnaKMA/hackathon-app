"""Configuration constants for the brain tumor detector."""

from pathlib import Path

# Model configuration (4-class multi-class classification)
MODEL_PATH = Path(__file__).parent / "models" / "brain_tumor_multiclass.keras"
MODEL_PATH_LEGACY = Path(__file__).parent / "models" / "brain_tumor_ft.keras"
IMG_SIZE = (224, 224)
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
NUM_CLASSES = len(CLASS_NAMES)

# API configuration
API_HOST = "127.0.0.1"
API_PORT = 8000

# Gemini API configuration
GEMINI_API_KEY = "AIzaSyDuHhCsGdPRq-ea4H3ENmyUc4WNnTAS2Uc"

# Class display information
CLASS_INFO = {
    "glioma": {"display": "Glioma", "type": "tumor", "description": "Brain tumor - Glioma"},
    "meningioma": {"display": "Meningioma", "type": "tumor", "description": "Brain tumor - Meningioma"},
    "notumor": {"display": "No Tumor", "type": "normal", "description": "No tumor detected"},
    "pituitary": {"display": "Pituitary", "type": "tumor", "description": "Brain tumor - Pituitary"}
}
