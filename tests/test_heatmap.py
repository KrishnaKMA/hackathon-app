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


@pytest.mark.parametrize("activation", [0.0, 0.2, 1.0])
def test_overlay_preserves_weak_regions_and_dimensions(monkeypatch, activation):
    from backend.services import heatmap_service

    monkeypatch.setattr(
        heatmap_service,
        "gradcam_values",
        lambda model, contents: np.full((7, 7), activation, dtype=np.float32),
    )
    source = io.BytesIO()
    Image.new("RGB", (32, 24), (100, 120, 140)).save(source, format="PNG")
    encoded = generate_heatmap_image(None, source.getvalue())
    overlay = Image.open(io.BytesIO(base64.b64decode(encoded.split(",")[1])))
    pixels = np.asarray(overlay)
    assert overlay.size == (32, 24)
    if activation <= 0.35:
        assert np.all(pixels == [100, 120, 140])
    else:
        assert np.all(pixels[..., 0] > 100)
        assert np.all(pixels[..., 1] < 120)
