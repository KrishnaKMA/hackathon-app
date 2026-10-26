"""Evaluate the entire labelled folder locally or via a deployed prediction API."""

import argparse
import hashlib
import json
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.config import CLASS_NAMES, MODEL_PATH
from backend.services.evaluation import classification_metrics, compare_results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, default=Path("test_images/kaggle_samples")
    )
    parser.add_argument("--output", type=Path, default=Path("reports/evaluations"))
    parser.add_argument(
        "--api-url",
        help="Explicitly evaluate a deployed backend instead of loading local weights",
    )
    parser.add_argument(
        "--model-version",
        default=None,
        help="Deployment/model version; required for API evaluations",
    )
    parser.add_argument("--compare", type=Path)
    args = parser.parse_args()
    if args.api_url and not args.model_version:
        parser.error("--model-version is required for deployed evaluation provenance")
    cases = []
    dataset_hash = hashlib.sha256()
    for name in CLASS_NAMES:
        directory = args.dataset / name
        files = sorted(
            (
                p
                for p in directory.rglob("*")
                if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
            ),
            key=lambda p: p.relative_to(args.dataset).as_posix().casefold(),
        )
        if not files:
            parser.error(f"No labelled images for {name} in {directory}")
        for path in files:
            relative = path.relative_to(args.dataset).as_posix()
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            dataset_hash.update(f"{relative}\0{name}\0{digest}\n".encode())
            cases.append((path, name, digest))
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    client = None
    if args.api_url:
        import httpx

        client = httpx.Client(base_url=args.api_url.rstrip("/"), timeout=120)
        readiness = client.get("/api/ready")
        readiness.raise_for_status()
        deployed_hash = readiness.json().get("model_sha256")
        if not isinstance(deployed_hash, str) or not re.fullmatch(
            r"[0-9a-f]{64}", deployed_hash
        ):
            client.close()
            raise ValueError(
                "Readiness must expose model_sha256; deploy the updated API before evaluation"
            )

        def predict(contents, filename):
            response = client.post("/api/predict", files={"file": (filename, contents)})
            response.raise_for_status()
            return response.json()

        model_info = {
            "version": args.model_version,
            "backend_url": args.api_url,
            "source": "deployed API; version supplied by deployment job",
            "sha256": deployed_hash,
        }
    else:
        from backend.model_loader import load_model
        from backend.services.ml_service import predict_scan

        model = load_model()

        def predict(contents, filename):
            return predict_scan(model, contents)

        model_info = {
            "version": args.model_version or MODEL_PATH.stem,
            "sha256": hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest(),
            "source": "local weights",
        }
    outcomes = []
    try:
        for index, (path, label, digest) in enumerate(cases):
            result = predict(path.read_bytes(), path.name)
            outcomes.append(
                {
                    "file": path.relative_to(args.dataset).as_posix(),
                    "sha256": digest,
                    "expected_class": label,
                    "predicted_class": result["predicted_class"],
                    "confidence": result["confidence"],
                }
            )
            if (index + 1) % 100 == 0:
                print(f"Evaluated {index + 1}/{len(cases)}", flush=True)
        if client:
            final_readiness = client.get("/api/ready")
            final_readiness.raise_for_status()
            if final_readiness.json().get("model_sha256") != deployed_hash:
                raise ValueError(
                    "Deployed model changed during evaluation; no result saved"
                )
    finally:
        if client:
            client.close()
    now = datetime.now(timezone.utc)
    run_id = now.strftime("%Y%m%dT%H%M%S%fZ")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True
        ).stdout.strip()
    )
    report = {
        "schema_version": 1,
        "run_id": run_id,
        "evaluated_utc": now.isoformat(),
        "model": model_info,
        "code": {"commit": commit, "working_tree_dirty": dirty},
        "python": platform.python_version(),
        "dataset": {
            "path": str(args.dataset),
            "sha256": dataset_hash.hexdigest(),
            "labels": "existing class directories, unchanged",
            "sample_count": len(cases),
            "provenance_warning": "Representativeness and training/test independence require an external audit; no clinical validation claimed",
        },
        "metrics": classification_metrics(
            [x["expected_class"] for x in outcomes],
            [x["predicted_class"] for x in outcomes],
            CLASS_NAMES,
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "cases": outcomes,
    }
    if args.compare:
        report["comparison"] = compare_results(
            report, json.loads(args.compare.read_text())
        )
    output = args.output / (run_id + ".json")
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    (args.output / "latest.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    metrics = report["metrics"]
    markdown = f"# NeuroDetect evaluation {run_id}\n\nDataset: {len(cases)} labelled images. Model: {model_info['version']}.\n\n"
    markdown += (
        "| Class | Precision | Recall | F1 | Support |\n|---|---:|---:|---:|---:|\n"
    )
    for name, values in metrics["per_class"].items():
        markdown += f"| {name} | {values['precision']:.4f} | {values['recall']:.4f} | {values['f1']:.4f} | {values['support']} |\n"
    markdown += f"\nMacro-F1: {metrics['macro_f1']:.4f}; weighted-F1: {metrics['weighted_f1']:.4f}; accuracy: {metrics['accuracy']:.4f}.\n\nConfusion matrix: rows=true, columns=predicted.\n\n"
    markdown += (
        "| Actual / predicted | "
        + " | ".join(CLASS_NAMES)
        + " |\n|---|"
        + "---:|" * len(CLASS_NAMES)
        + "\n"
    )
    for name, row in zip(CLASS_NAMES, metrics["confusion_matrix"], strict=True):
        markdown += "| " + name + " | " + " | ".join(map(str, row)) + " |\n"
    markdown += "\nThese are dataset-specific results, not clinical performance estimates. The 11 unlabelled user images are excluded.\n"
    if "comparison" in report:
        markdown += (
            "\nComparison with previous evaluation:\n```json\n"
            + json.dumps(report["comparison"], indent=2)
            + "\n```\n"
        )
    output.with_suffix(".md").write_text(markdown, encoding="utf-8")
    (args.output / "latest.md").write_text(markdown, encoding="utf-8")
    print(json.dumps({"output": str(output), "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
