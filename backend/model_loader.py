"""Model loading and management."""

import logging
from pathlib import Path
import tensorflow as tf

from config import MODEL_PATH

logging.basicConfig(level=logging.INFO)


def load_model():
    """
    Load the trained tumor detection model.
    Azure-safe: validates path and logs clearly.
    """
    model_path = Path(MODEL_PATH)

    logging.info(f"Loading model from: {model_path}")

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found at {model_path}. "
            "Ensure the models/ directory is included in the deployment ZIP."
        )

    try:
        model = tf.keras.models.load_model(model_path)
        logging.info("Model loaded successfully")
        return model
    except Exception as e:
        logging.exception("Failed to load model")
        raise
