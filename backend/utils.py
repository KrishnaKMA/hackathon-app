"""Utility functions for image processing and model interpretation."""

import io

import numpy as np
from PIL import Image

from .config import IMG_SIZE


def preprocess_image(img_bytes):
    """
    Load and preprocess image bytes to model input format.

    Args:
        img_bytes: Raw image bytes

    Returns:
        Preprocessed image array in shape (1, 224, 224, 3) with [-1, 1] range
    """
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)

    # Normalize to [0, 1]
    x = np.array(img, dtype=np.float32) / 255.0
    x = np.expand_dims(x, axis=0)  # (1, 224, 224, 3)

    # MobileNetV2 expects [-1, 1]; convert from [0, 1]
    x = (x * 2.0) - 1.0

    return x
