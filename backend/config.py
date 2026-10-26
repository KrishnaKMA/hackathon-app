"""Configuration constants for the brain tumor detector."""

import os
from pathlib import Path

# Model configuration
MODEL_PATH = Path(
    os.getenv("MODEL_PATH")
    or Path(__file__).parent / "models" / "brain_tumor_multiclass.keras"
)
MODEL_PATH_LEGACY = Path(__file__).parent / "models" / "brain_tumor_ft.keras"

IMG_SIZE = (224, 224)

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
NUM_CLASSES = len(CLASS_NAMES)


# API configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))


# Temporary chat provider; keys remain on the backend.
CHAT_API_KEY = os.getenv("GROQ_API_KEY", "")
CHAT_MODEL = os.getenv("GROQ_MODEL", os.getenv("CHAT_MODEL", "openai/gpt-oss-20b"))
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")
CHAT_API_URL = "https://api.groq.com/openai/v1/chat/completions"
MAX_IMAGE_PIXELS = int(os.getenv("MAX_IMAGE_PIXELS", "4000000"))
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "10")) * 1024 * 1024
TF_INTRA_THREADS = int(os.getenv("TF_INTRA_THREADS", "4"))
TF_INTER_THREADS = int(os.getenv("TF_INTER_THREADS", "1"))
INFERENCE_MODE = os.getenv("INFERENCE_MODE", "compiled")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
MODEL_BACKEND = os.getenv("MODEL_BACKEND", "keras")
CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "ALLOWED_ORIGINS",
        os.getenv(
            "CORS_ORIGINS",
            "http://localhost:5500,http://127.0.0.1:5500,https://frontend-4s5x.onrender.com",
        ),
    ).split(",")
    if origin.strip()
]

# Class display information
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
