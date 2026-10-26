"""Float32 LiteRT inference and analytic Grad-CAM using original trained head weights."""

import hashlib
import json
from pathlib import Path

import numpy as np

from ..config import MODEL_PATH


class LiteModel:
    backend = "tflite"

    def __init__(self):
        from ai_edge_litert.interpreter import Interpreter

        root = Path(__file__).resolve().parents[1] / "models"
        metadata = json.loads((root / "lite_metadata.json").read_text())
        path = root / "brain_tumor_float32.tflite"
        if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["tflite_sha256"]:
            raise ValueError("Lite model checksum mismatch")
        if (
            hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()
            != metadata["source_sha256"]
        ):
            raise ValueError("Source weights changed; re-export the Lite artifacts")
        if (
            hashlib.sha256((root / "gradcam_head.npz").read_bytes()).hexdigest()
            != metadata["head_sha256"]
        ):
            raise ValueError("Grad-CAM parameter checksum mismatch")
        self._neurodetect_sha256 = metadata["source_sha256"]
        self.runtime_sha256 = metadata["tflite_sha256"]
        self.interpreter = Interpreter(model_path=str(path), num_threads=1)
        self.interpreter.allocate_tensors()
        self.input_index = self.interpreter.get_input_details()[0]["index"]
        outputs = self.interpreter.get_output_details()
        self.prob_index = next(
            item["index"] for item in outputs if tuple(item["shape"]) == (1, 4)
        )
        self.conv_index = next(
            item["index"] for item in outputs if len(item["shape"]) == 4
        )
        with np.load(root / "gradcam_head.npz", allow_pickle=False) as archive:
            self.params = {key: archive[key] for key in archive.files}
        if str(self.params["source_sha256"]) != self._neurodetect_sha256:
            raise ValueError("Grad-CAM parameters do not match the source model")

    def infer(self, x):
        self.interpreter.set_tensor(self.input_index, np.asarray(x, dtype=np.float32))
        self.interpreter.invoke()
        return self.interpreter.get_tensor(
            self.prob_index
        ), self.interpreter.get_tensor(self.conv_index)

    def __call__(self, x, training=False):
        return self.infer(x)[0]

    def predict(self, x, verbose=0):
        return self(x)

    def _neurodetect_infer(self, x):
        return self(x)

    def gradcam(self, x):
        probabilities, activations = self.infer(x)
        conv = activations[0]
        p = self.params
        scale = p["gamma"] / np.sqrt(p["variance"] + np.float32(p["epsilon"]))
        normalized = (conv - p["mean"]) * scale + p["beta"]
        pooled = np.clip(normalized, 0, 6).mean(axis=(0, 1))
        z1 = pooled @ p["w0"] + p["b0"]
        z2 = np.maximum(z1, 0) @ p["w1"] + p["b1"]
        index = int(np.argmax(probabilities[0]))
        gradient = p["w0"] @ ((z1 > 0) * (p["w1"] @ ((z2 > 0) * p["w2"][:, index])))
        gradient = (
            gradient
            * scale
            * (normalized > 0)
            * (normalized < 6)
            / (conv.shape[0] * conv.shape[1])
        )
        weights = gradient.mean(axis=(0, 1))
        cam = np.maximum(np.sum(conv * weights, axis=-1), 0)
        if not np.isfinite(cam).all():
            raise ValueError("Non-finite Grad-CAM values")
        maximum = float(cam.max())
        return cam / maximum if maximum > 0 else np.zeros_like(cam)
