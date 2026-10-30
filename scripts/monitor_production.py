"""Deploy a tested revision or check production without local ML dependencies."""

import argparse
import base64
import io
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.schemas.prediction import HeatmapResponse, PredictionResponse  # noqa: E402


def trigger_deploy(client, hook, revision):
    if not revision or len(revision) != 40:
        raise ValueError("A full tested commit SHA is required")
    parts = urlsplit(hook)
    if parts.scheme != "https" or parts.hostname != "api.render.com":
        raise ValueError("Configure the RENDER_DEPLOY_HOOK_URL repository secret")
    # Never print the hook or an HTTP exception containing its secret query string.
    try:
        response = client.post(httpx.URL(hook).copy_merge_params({"ref": revision}))
    except httpx.HTTPError:
        raise RuntimeError("Render deployment request failed") from None
    if response.status_code not in (200, 202):
        raise RuntimeError(f"Render deployment returned HTTP {response.status_code}")


def wait_ready(client, revision=None, timeout=240, interval=15):
    deadline = time.monotonic() + timeout
    while True:
        try:
            response = client.get("/api/ready")
            response.raise_for_status()
            ready = response.json()
            if ready.get("status") == "ready" and ready.get("model_sha256"):
                if not revision or ready.get("revision") == revision:
                    return ready
        except (httpx.HTTPError, ValueError):
            pass
        if time.monotonic() >= deadline:
            raise RuntimeError(
                "Production did not become ready at the expected revision"
            )
        time.sleep(interval)


def check_production(client):
    fixture = ROOT / "tests/fixtures/user_cases/1.webp"
    data = fixture.read_bytes()
    timings = []
    for _ in range(3):
        started = time.perf_counter()
        response = client.post(
            "/api/predict", files={"file": (fixture.name, data, "image/webp")}
        )
        response.raise_for_status()
        PredictionResponse.model_validate(response.json())
        timings.append(
            {
                "http_ms": round((time.perf_counter() - started) * 1000, 2),
                "server_timing": response.headers.get("server-timing"),
            }
        )
    response = client.post(
        "/api/heatmap", files={"file": (fixture.name, data, "image/webp")}
    )
    response.raise_for_status()
    heatmap = HeatmapResponse.model_validate(response.json()).heatmap
    with Image.open(io.BytesIO(base64.b64decode(heatmap.split(",", 1)[1]))) as image:
        image.load()
        with Image.open(fixture) as original:
            if image.size != original.size:
                raise RuntimeError("Heatmap dimensions differ from the input")
    response = client.get("/ui/")
    response.raise_for_status()
    if "NeuroDetect" not in response.text:
        raise RuntimeError("Frontend did not contain the expected app")
    invalid = client.post("/api/predict", files={"file": ("invalid.png", b"invalid")})
    if invalid.status_code != 400:
        raise RuntimeError("Invalid image validation failed")
    return timings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deploy", action="store_true")
    args = parser.parse_args()
    report = {"checked_utc": datetime.now(timezone.utc).isoformat(), "ok": False}
    output = ROOT / "reports/production-check.json"
    try:
        base = os.getenv("PRODUCTION_URL", "https://hackathon-app-adb9.onrender.com")
        revision = os.getenv("EXPECTED_REVISION") if args.deploy else None
        with httpx.Client(base_url=base, timeout=90) as client:
            if args.deploy:
                trigger_deploy(
                    client, os.getenv("RENDER_DEPLOY_HOOK_URL", ""), revision
                )
            report["readiness"] = wait_ready(
                client, revision, timeout=1200 if args.deploy else 240
            )
            report["prediction_timings"] = check_production(client)
            final = wait_ready(client, revision)
            if final != report["readiness"]:
                raise RuntimeError("Deployment changed during verification")
            report["ok"] = True
        print("Production readiness, revision, predictions, heatmap and UI passed")
    except Exception as error:
        # Avoid leaking deployment credentials from exception URLs.
        report["error"] = type(error).__name__
        print(f"Production verification failed ({type(error).__name__})")
        raise SystemExit(1) from None
    finally:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
