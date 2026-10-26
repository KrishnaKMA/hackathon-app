"""Measure local or deployed API latency with real fixture uploads."""

import argparse
import base64
import io
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
import numpy as np
from PIL import Image

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--api-url", default="http://127.0.0.1:8000")
parser.add_argument("--runs", type=int, default=30)
parser.add_argument(
    "--output", type=Path, default=Path("reports/endpoint-benchmark.json")
)
parser.add_argument("--expected-model-sha")
args = parser.parse_args()
if args.runs < 1:
    parser.error("--runs must be positive")
root = Path("tests/fixtures/user_cases")
args.output.parent.mkdir(parents=True, exist_ok=True)
report = {
    "base_url": args.api_url.rstrip("/"),
    "runs": args.runs,
    "timings": {},
    "cases": [],
    "measured_utc": datetime.now(timezone.utc).isoformat(),
}
samples = {}
with httpx.Client(base_url=report["base_url"], timeout=90) as client:
    first_request = time.perf_counter()
    readiness = client.get("/api/ready")
    report["first_readiness_request_ms"] = (time.perf_counter() - first_request) * 1000
    readiness.raise_for_status()
    report["model_sha256"] = readiness.json().get("model_sha256")
    if args.expected_model_sha and report["model_sha256"] != args.expected_model_sha:
        raise ValueError("Deployed model hash differs from expected weights")
    if not report["model_sha256"]:
        raise ValueError(
            "Updated API is not deployed: readiness must expose model_sha256"
        )
    data = (root / "1.webp").read_bytes()
    for i in range(args.runs + 3):
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
            image.save(args.output.parent / (args.output.stem + "-gradcam.png"))
        report["cases"].append(
            {
                "file": path.name,
                "prediction": result.json(),
                "heatmap_status": heat.status_code,
                "heatmap_dimensions": image.size,
            }
        )
    samples["heatmap_http_roundtrip"] = heat_times
    report["health"] = client.get("/api/health").json()
    invalid = client.post("/api/predict", files={"file": ("invalid.png", b"invalid")})
    report["invalid_upload_status"] = invalid.status_code
    assert invalid.status_code == 400
    allowed = client.options(
        "/api/predict",
        headers={
            "Origin": "https://frontend-4s5x.onrender.com",
            "Access-Control-Request-Method": "POST",
        },
    )
    report["hosted_cors_status"] = allowed.status_code
    assert allowed.status_code == 200
    started = time.perf_counter()
    chat = client.post(
        "/api/chat", json={"message": "What does the model confidence score mean?"}
    )
    report["chat_roundtrip_ms"] = (time.perf_counter() - started) * 1000
    chat.raise_for_status()
    report["chat"] = chat.json()
    final_ready = client.get("/api/ready")
    final_ready.raise_for_status()
    if final_ready.json().get("model_sha256") != report["model_sha256"]:
        raise ValueError("Model changed while timing requests; report discarded")
for name, values in samples.items():
    report["timings"][name] = {
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.percentile(values, 95)),
        "n": len(values),
    }
args.output.write_text(json.dumps(report, indent=2))
print(json.dumps(report["timings"], indent=2))
