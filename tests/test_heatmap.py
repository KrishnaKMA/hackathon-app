import base64
import io

import numpy as np
import pytest
from PIL import Image

from backend.services.heatmap_service import generate_heatmap_image, gradcam_values


@pytest.mark.integration
@pytest.mark.parametrize("number", range(1, 12))
def test_gradcam_real_model(real_model, user_cases, number):
    root, _ = user_cases
    contents = (root / f"{number}.webp").read_bytes()
    values = gradcam_values(real_model, contents)
    assert values.ndim == 2
    assert np.isfinite(values).all()
    assert values.min() >= 0 and values.max() <= 1
    encoded = generate_heatmap_image(real_model, contents)
    assert encoded.startswith("data:image/png;base64,")
    overlay = Image.open(io.BytesIO(base64.b64decode(encoded.split(",")[1])))
    assert overlay.size == Image.open(io.BytesIO(contents)).size
    assert overlay.format == "PNG"
