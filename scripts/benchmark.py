"""Isolated-process inference/thread/TFLite benchmark. Never a CI latency gate."""

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import psutil

parser = argparse.ArgumentParser()
parser.add_argument("--threads", type=int, default=0)
parser.add_argument("--runs", type=int, default=30)
parser.add_argument("--lite", action="store_true")
args = parser.parse_args()
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

if args.threads:
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    tf.config.threading.set_inter_op_parallelism_threads(1)
from backend.config import CLASS_NAMES, MODEL_PATH
from backend.utils import preprocess_image

out = Path("reports")
out.mkdir(exist_ok=True)
process = psutil.Process()
report = {
    "platform": platform.platform(),
    "cpu": platform.processor(),
    "logical_cpus": os.cpu_count(),
    "python": platform.python_version(),
    "tensorflow": tf.__version__,
    "threads": args.threads,
    "inter_threads": 1 if args.threads else "default",
    "runs": args.runs,
    "configurations": {},
}
files = sorted(
    Path("tests/fixtures/user_cases").glob("*.webp"), key=lambda p: int(p.stem)
)
images = [preprocess_image(p.read_bytes()) for p in files]
x = images[0]
t = time.perf_counter()
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
report["load_ms"] = (time.perf_counter() - t) * 1000


def measure(fn, runs=args.runs):
    samples = []
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000)
    return {
        "median_ms": float(np.median(samples)),
        "p95_ms": float(np.percentile(samples, 95)),
        "n": runs,
    }


start = time.perf_counter()
model.predict(x, verbose=0)
report["cold_predict_ms"] = (time.perf_counter() - start) * 1000
for _ in range(3):
    model(x, training=False).numpy()
compiled = tf.function(
    lambda im: model(im, training=False),
    input_signature=[tf.TensorSpec((1, 224, 224, 3), tf.float32)],
)
for _ in range(3):
    compiled(x).numpy()
baseline = np.concatenate([model.predict(im, verbose=0) for im in images])
for name, fn in [
    ("original_predict", lambda im: model.predict(im, verbose=0)),
    ("direct_keras", lambda im: model(im, training=False).numpy()),
    ("compiled_keras", lambda im: compiled(im).numpy()),
]:
    report["configurations"][name] = measure(lambda: fn(x))
    outputs = np.concatenate([fn(im) for im in images])
    report["configurations"][name].update(
        {
            "max_probability_delta": float(np.max(np.abs(outputs - baseline))),
            "class_agreement": float(np.mean(outputs.argmax(1) == baseline.argmax(1))),
            "rss_mib": process.memory_info().rss / 1024**2,
        }
    )
report["preprocessing"] = measure(lambda: preprocess_image(files[0].read_bytes()))
report["user_case_outputs"] = [
    {
        "file": p.name,
        "predicted_class": CLASS_NAMES[int(prob.argmax())],
        "probabilities": prob.tolist(),
        "expected_class": None,
    }
    for p, prob in zip(files, baseline)
]
# Directory labels are preserved independently from unlabelled user cases.
labelled = [
    (p, cls)
    for cls in CLASS_NAMES
    for p in sorted((Path("test_images/kaggle_samples") / cls).glob("*.jpg"))[:10]
]
labelled_x = [preprocess_image(p.read_bytes()) for p, _ in labelled]
labelled_baseline = np.concatenate(
    [model(im, training=False).numpy() for im in labelled_x]
)
expected = np.array([CLASS_NAMES.index(cls) for _, cls in labelled])
report["labelled_regression"] = {
    "n": len(labelled),
    "correct": int(np.sum(labelled_baseline.argmax(1) == expected)),
    "accuracy": float(np.mean(labelled_baseline.argmax(1) == expected)),
    "files": [str(p) for p, _ in labelled],
}
if args.lite:
    for variant in ["float32", "dynamic_range"]:
        try:
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            if variant == "dynamic_range":
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converted = converter.convert()
            (out / f"model-{variant}.tflite").write_bytes(converted)
            interpreter = tf.lite.Interpreter(
                model_content=converted, num_threads=args.threads or 1
            )
            interpreter.allocate_tensors()
            inp = interpreter.get_input_details()[0]["index"]
            output = interpreter.get_output_details()[0]["index"]

            def invoke(im):
                interpreter.set_tensor(inp, im)
                interpreter.invoke()
                return interpreter.get_tensor(output)

            for _ in range(3):
                invoke(x)
            outputs = np.concatenate([invoke(im) for im in images])
            labelled_outputs = np.concatenate([invoke(im) for im in labelled_x])
            result = measure(lambda: invoke(x))
            result.update(
                {
                    "model_bytes": len(converted),
                    "rss_mib": process.memory_info().rss / 1024**2,
                    "max_probability_delta": float(np.max(np.abs(outputs - baseline))),
                    "class_agreement": float(
                        np.mean(outputs.argmax(1) == baseline.argmax(1))
                    ),
                    "labelled_accuracy": float(
                        np.mean(labelled_outputs.argmax(1) == expected)
                    ),
                }
            )
            report["configurations"]["tflite_" + variant] = result
        except Exception as exc:
            report["configurations"]["tflite_" + variant] = {
                "conversion_error": str(exc)
            }
report["peak_rss_mib"] = (
    process.memory_info().peak_wset / 1024**2
    if hasattr(process.memory_info(), "peak_wset")
    else None
)
(out / f"benchmark-threads-{args.threads}.json").write_text(
    json.dumps(report, indent=2)
)
print(json.dumps(report, indent=2))
