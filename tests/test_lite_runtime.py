import numpy as np
import pytest

from backend.services.heatmap_service import gradcam_values
from backend.utils import preprocess_image


@pytest.fixture(scope="module")
def lite_model():
    pytest.importorskip("ai_edge_litert")
    from backend.services.lite_runtime import LiteModel

    return LiteModel()


@pytest.mark.integration
@pytest.mark.parametrize("number", range(1, 12))
def test_lite_predictions_and_true_gradcam_match_keras(
    real_model, lite_model, user_cases, number
):
    root, _ = user_cases
    data = (root / f"{number}.webp").read_bytes()
    x = preprocess_image(data)
    expected = np.asarray(real_model(x, training=False))
    actual = lite_model.predict(x)
    assert actual.argmax() == expected.argmax()
    np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=1e-3)
    reference_cam = gradcam_values(real_model, data)
    lite_cam = gradcam_values(lite_model, data)
    np.testing.assert_allclose(lite_cam, reference_cam, atol=0.005, rtol=0.02)
    assert lite_model._neurodetect_sha256 == real_model._neurodetect_sha256


@pytest.mark.integration
@pytest.mark.parametrize("name", ["glioma", "meningioma", "notumor", "pituitary"])
def test_lite_gradcam_covers_each_class(real_model, lite_model, name):
    import json
    from pathlib import Path

    cases = json.loads(Path("tests/fixtures/labelled-regression.json").read_text())
    path = Path(next(case["file"] for case in cases if case["expected_class"] == name))
    data = path.read_bytes()
    assert int(lite_model.predict(preprocess_image(data)).argmax()) == [
        "glioma",
        "meningioma",
        "notumor",
        "pituitary",
    ].index(name)
    np.testing.assert_allclose(
        gradcam_values(lite_model, data),
        gradcam_values(real_model, data),
        atol=0.005,
        rtol=0.02,
    )
