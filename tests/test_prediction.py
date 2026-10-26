import io
import json

import numpy as np
import pytest
from PIL import Image

from backend.config import CLASS_NAMES
from backend.services.ml_service import predict_scan
from backend.utils import preprocess_image


@pytest.mark.parametrize(
    "color,expected", [(0, -1.0), (255, 1.0), (128, 128 / 255 * 2 - 1)]
)
def test_preprocessing_training_contract(color, expected):
    stream = io.BytesIO()
    Image.new("RGB", (19, 27), (color, color, color)).save(stream, format="PNG")
    x = preprocess_image(stream.getvalue())
    assert x.shape == (1, 224, 224, 3)
    assert x.dtype == np.float32
    np.testing.assert_allclose(x, expected, atol=1e-7)


@pytest.mark.integration
@pytest.mark.parametrize("number", range(1, 12))
def test_all_user_cases(real_model, user_cases, number):
    root, cases = user_cases
    case = next(c for c in cases if c["file"] == f"{number}.webp")
    result = predict_scan(real_model, (root / case["file"]).read_bytes())
    assert result["predicted_class"] in CLASS_NAMES
    assert set(result["all_predictions"]) == set(CLASS_NAMES)
    assert 0 <= result["confidence"] <= 1
    assert sum(
        x["probability"] for x in result["all_predictions"].values()
    ) == pytest.approx(1, abs=0.001)
    if case["expected_class"] is not None:
        assert result["predicted_class"] == case["expected_class"]


@pytest.mark.integration
def test_keras_implementations_agree(real_model, user_cases):
    root, cases = user_cases
    for case in cases:
        x = preprocess_image((root / case["file"]).read_bytes())
        baseline = real_model.predict(x, verbose=0)
        direct = real_model(x, training=False).numpy()
        np.testing.assert_allclose(baseline, direct, atol=1e-4, rtol=1e-4)
        assert baseline.argmax() == direct.argmax()


@pytest.mark.integration
def test_deployed_regression_reference(real_model, user_cases):
    from pathlib import Path

    path = Path("tests/fixtures/user_cases/deployed_predictions.json")
    if not path.exists():
        pytest.skip("Deployed reference not captured")
    refs = json.loads(path.read_text())["cases"]
    successful = [item for item in refs if "prediction" in item]
    if not successful:
        pytest.skip("Deployed backend unavailable; no reference predictions")
    root, _ = user_cases
    for ref in successful:
        actual = predict_scan(real_model, (root / ref["file"]).read_bytes())
        assert actual["predicted_class"] == ref["prediction"]["predicted_class"]
        assert actual["confidence"] == pytest.approx(
            ref["prediction"]["confidence"], abs=1e-3
        )


@pytest.mark.integration
def test_labelled_dataset_regression(real_model):
    import json
    from pathlib import Path

    outcomes = []
    cases = json.loads(Path("tests/fixtures/labelled-regression.json").read_text())
    for case in cases:
        path = Path(case["file"])
        actual = predict_scan(real_model, path.read_bytes())["predicted_class"]
        outcomes.append(
            {
                "file": case["file"],
                "expected_class": case["expected_class"],
                "predicted_class": actual,
                "correct": actual == case["expected_class"],
            }
        )
    report = {
        "source": "Fixed original 40-file regression manifest; labels unchanged",
        "n": len(outcomes),
        "correct": sum(item["correct"] for item in outcomes),
        "cases": outcomes,
    }
    Path("reports").mkdir(exist_ok=True)
    Path("reports/labelled-regression.json").write_text(json.dumps(report, indent=2))
    # Baseline measured 39/40 before selecting the optimized runtime.
    assert report["correct"] >= 39


def test_labelled_manifest_has_fixed_class_coverage():
    import json
    from pathlib import Path

    cases = json.loads(Path("tests/fixtures/labelled-regression.json").read_text())
    assert len({case["file"] for case in cases}) == 40
    for name in CLASS_NAMES:
        assert sum(case["expected_class"] == name for case in cases) == 10
    assert all(Path(case["file"]).is_file() for case in cases)
