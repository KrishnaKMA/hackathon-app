"""Model loading and management."""

import tensorflow as tf
from .config import MODEL_PATH


def load_model():
    """Load the trained tumor detection model."""
    model = tf.keras.models.load_model(MODEL_PATH)
    return model
