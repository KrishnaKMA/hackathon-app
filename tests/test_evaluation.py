import pytest

from backend.services.evaluation import classification_metrics, compare_results


def test_metrics_known_confusion_matrix():
    m = classification_metrics(
        ["a", "a", "b", "c"], ["a", "b", "b", "a"], ["a", "b", "c", "d"]
    )
    assert m["confusion_matrix"] == [
        [1, 1, 0, 0],
        [0, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 0],
    ]
    assert m["per_class"]["a"]["precision"] == 0.5
    assert m["per_class"]["a"]["recall"] == 0.5
    assert m["per_class"]["b"]["f1"] == pytest.approx(2 / 3)
    assert m["macro_f1"] == pytest.approx((0.5 + 2 / 3) / 4)
    assert m["weighted_f1"] == pytest.approx((2 * 0.5 + 2 / 3) / 4)
    assert m["missing_classes"] == ["d"]


@pytest.mark.parametrize("truth,pred", [([], []), (["a"], []), (["x"], ["a"])])
def test_metrics_reject_bad_inputs(truth, pred):
    with pytest.raises(ValueError):
        classification_metrics(truth, pred, ["a"])


def test_comparison_rejects_changed_dataset():
    result = {"dataset": {"sha256": "one"}, "metrics": {}}
    with pytest.raises(ValueError):
        compare_results(result, {"dataset": {"sha256": "two"}})


def test_saved_comparison_delta():
    metrics = classification_metrics(["a", "b"], ["a", "b"], ["a", "b"])
    old = {
        "run_id": "old",
        "model": {"version": "v1"},
        "dataset": {"sha256": "same"},
        "metrics": metrics,
    }
    new = {**old, "run_id": "new", "model": {"version": "v2"}}
    assert compare_results(new, old)["delta"] == {
        "accuracy": 0,
        "macro_f1": 0,
        "weighted_f1": 0,
    }


@pytest.mark.parametrize("changed", [False, True])
def test_hosted_evaluation_records_hash_and_rejects_midrun_change(
    tmp_path, monkeypatch, changed
):
    import json
    import sys

    import httpx
    from PIL import Image

    from backend.config import CLASS_NAMES
    from scripts.evaluate import main

    dataset = tmp_path / "dataset"
    output = tmp_path / "output"
    for name in CLASS_NAMES:
        folder = dataset / name
        folder.mkdir(parents=True)
        Image.new("RGB", (2, 2)).save(folder / (name + ".png"))
    calls = 0

    def handler(request):
        nonlocal calls
        if request.url.path == "/api/ready":
            calls += 1
            return httpx.Response(
                200,
                json={
                    "status": "ready",
                    "model_sha256": ("b" if changed and calls > 1 else "a") * 64,
                },
            )
        name = next(
            name
            for name in CLASS_NAMES
            if f'filename="{name}.png"'.encode() in request.content
        )
        return httpx.Response(200, json={"predicted_class": name, "confidence": 0.9})

    real = httpx.Client
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kwargs: real(transport=httpx.MockTransport(handler), **kwargs),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate",
            "--dataset",
            str(dataset),
            "--output",
            str(output),
            "--api-url",
            "https://example.invalid",
            "--model-version",
            "v2",
        ],
    )
    if changed:
        with pytest.raises(ValueError, match="changed during"):
            main()
        assert not (output / "latest.json").exists()
    else:
        main()
        saved = json.loads((output / "latest.json").read_text())
        assert saved["model"]["sha256"] == "a" * 64
        assert saved["metrics"]["macro_f1"] == 1
        assert saved["dataset"]["sample_count"] == 4
        assert (output / "latest.md").exists()
