"""Exercise render.yaml's startup command and real endpoints in a fresh process."""

import base64
import io
import json
import os
import shlex
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx
import psutil
import yaml
from PIL import Image


def main():
    root = Path(__file__).resolve().parents[1]
    service = yaml.safe_load((root / "render.yaml").read_text())["services"][0]
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    command = shlex.split(service["startCommand"].replace("$PORT", str(port)))
    assert command[0] == "uvicorn"
    command = [sys.executable, "-m", "uvicorn", *command[1:]]
    environment = os.environ.copy()
    environment.update(
        {
            item["key"]: str(item["value"])
            for item in service["envVars"]
            if "value" in item
        }
    )
    environment.pop("GROQ_API_KEY", None)
    environment["PORT"] = str(port)
    result = {
        "start_command": service["startCommand"],
        "python": sys.version,
        "observed_loading": False,
    }
    with tempfile.TemporaryFile(mode="w+") as log:
        process = subprocess.Popen(
            command, cwd=root, env=environment, stdout=log, stderr=log
        )
        try:
            with httpx.Client(
                base_url=f"http://127.0.0.1:{port}", timeout=20
            ) as client:
                deadline = time.monotonic() + 120
                while time.monotonic() < deadline:
                    if process.poll() is not None:
                        raise RuntimeError("API exited during startup")
                    try:
                        health = client.get("/api/health")
                        health.raise_for_status()
                        result["observed_loading"] |= (
                            health.json()["model_status"] == "loading"
                        )
                        readiness = client.get(service["healthCheckPath"])
                        if readiness.status_code == 200:
                            break
                        assert readiness.status_code == 503
                    except httpx.ConnectError:
                        pass
                    time.sleep(0.2)
                else:
                    raise RuntimeError("Readiness timed out")
                assert len(readiness.json()["model_sha256"]) == 64
                result["model_sha256"] = readiness.json()["model_sha256"]
                fixture = root / "tests/fixtures/user_cases/1.webp"

                def upload(endpoint, contents):
                    return client.post(
                        endpoint, files={"file": ("1.webp", contents, "image/webp")}
                    )

                prediction = upload("/api/predict", fixture.read_bytes())
                prediction.raise_for_status()
                assert len(prediction.json()["all_predictions"]) == 4
                result["predicted_class"] = prediction.json()["predicted_class"]
                heat = upload("/api/heatmap", fixture.read_bytes())
                heat.raise_for_status()
                overlay = Image.open(
                    io.BytesIO(base64.b64decode(heat.json()["heatmap"].split(",")[1]))
                )
                assert overlay.size == Image.open(fixture).size
                assert upload("/api/predict", b"invalid").status_code == 400
                assert (
                    client.post(
                        "/api/chat", json={"message": "What is confidence?"}
                    ).json()["error"]
                    == "missing_api_key"
                )
                allowed = client.options(
                    "/api/predict",
                    headers={
                        "Origin": "https://frontend-4s5x.onrender.com",
                        "Access-Control-Request-Method": "POST",
                    },
                )
                assert allowed.status_code == 200
                assert (
                    allowed.headers["access-control-allow-origin"]
                    == "https://frontend-4s5x.onrender.com"
                )
                blocked = client.options(
                    "/api/predict",
                    headers={
                        "Origin": "https://untrusted.invalid",
                        "Access-Control-Request-Method": "POST",
                    },
                )
                assert blocked.status_code == 400
                memory = psutil.Process(process.pid).memory_info()
                result["rss_mib"] = memory.rss / 1024**2
                if sys.platform.startswith("linux"):
                    for line in (
                        Path(f"/proc/{process.pid}/status").read_text().splitlines()
                    ):
                        if line.startswith("VmHWM:"):
                            result["peak_rss_mib"] = int(line.split()[1]) / 1024
                elif hasattr(memory, "peak_wset"):
                    result["peak_rss_mib"] = memory.peak_wset / 1024**2
                result.update(
                    {
                        "ready": True,
                        "prediction": True,
                        "heatmap": True,
                        "invalid_upload": True,
                        "chat_without_key": True,
                        "hosted_cors": True,
                    }
                )
        except BaseException:
            log.seek(0)
            print(log.read()[-6000:], file=sys.stderr)
            raise
        finally:
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
    output = root / "reports" / ("deployment-smoke-" + sys.platform + ".json")
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
