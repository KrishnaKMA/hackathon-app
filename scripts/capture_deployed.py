"""Capture deployed model predictions as regression references, never clinical labels."""

import json
import re
from datetime import UTC, datetime
from pathlib import Path

import httpx

folder = Path("tests/fixtures/user_cases")
manifest = json.loads((folder / "manifest.json").read_text())
url = "https://frontend-4s5x.onrender.com"
report = {"frontend": url, "captured_utc": datetime.now(UTC).isoformat(), "cases": []}
with httpx.Client(
    timeout=httpx.Timeout(120.0, connect=15.0), follow_redirects=True
) as client:
    source = client.get(url + "/js/api.js")
    source.raise_for_status()
    backend = re.search(r'"(https://[^"\s]+onrender.com)"', source.text).group(1)
    report["backend"] = backend
    for case in manifest["cases"]:
        path = folder / case["file"]
        item = {"file": path.name, "sha256": case["sha256"]}
        try:
            response = client.post(
                backend + "/predict",
                files={"file": (path.name, path.read_bytes(), "image/webp")},
            )
            item["http_status"] = response.status_code
            if response.is_success:
                item["prediction"] = response.json()
            else:
                item["error"] = "Deployed endpoint unavailable"
        except httpx.HTTPError as exc:
            item["error"] = type(exc).__name__
        report["cases"].append(item)
        print(
            path.name,
            item.get("http_status"),
            item.get("prediction", {}).get("predicted_class"),
            flush=True,
        )
        Path("reports/deployed-reference.json").write_text(json.dumps(report, indent=2))
