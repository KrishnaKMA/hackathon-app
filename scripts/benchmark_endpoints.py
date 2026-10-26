"""Measure the running local API with real fixture uploads."""

import base64
import io
import json
import time
from pathlib import Path

import httpx
import numpy as np
from PIL import Image

root = Path("tests/fixtures/user_cases")
report = {"base_url": "http://127.0.0.1:8000", "runs": 30, "timings": {}, "cases": []}
samples = {}
with httpx.Client(base_url=report["base_url"], timeout=90) as client:
    client.get("/api/ready").raise_for_status()
    data = (root / "1.webp").read_bytes()
    for i in range(33):
        start = time.perf_counter()
        response = client.post(
            "/api/predict", files={"file": ("1.webp", data, "image/webp")}
        )
        response.raise_for_status()
        elapsed = (time.perf_counter() - start) * 1000
        if i < 3:
            continue
        samples.setdefault("http_roundtrip", []).append(elapsed)
        for metric in response.headers["server-timing"].split(", "):
            name, value = metric.split(";dur=")
            samples.setdefault(name, []).append(float(value))
    heat_times = []
    for path in sorted(root.glob("*.webp"), key=lambda p: int(p.stem)):
        result = client.post(
            "/api/predict", files={"file": (path.name, path.read_bytes(), "image/webp")}
        )
        result.raise_for_status()
        start = time.perf_counter()
        heat = client.post(
            "/api/heatmap", files={"file": (path.name, path.read_bytes(), "image/webp")}
        )
        heat.raise_for_status()
        heat_times.append((time.perf_counter() - start) * 1000)
        image = Image.open(
            io.BytesIO(base64.b64decode(heat.json()["heatmap"].split(",")[1]))
        )
        assert image.size == Image.open(path).size
        if path.name == "1.webp":
            image.save("reports/gradcam-example.png")
        report["cases"].append(
            {
                "file": path.name,
                "prediction": result.json(),
                "heatmap_status": heat.status_code,
                "heatmap_dimensions": image.size,
            }
        )
    samples["heatmap_http_roundtrip"] = heat_times
for name, values in samples.items():
    report["timings"][name] = {
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.percentile(values, 95)),
        "n": len(values),
    }
Path("reports/endpoint-benchmark.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report["timings"], indent=2))
