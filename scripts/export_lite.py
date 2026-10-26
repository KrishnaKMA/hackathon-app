"""Export original trained weights for a low-memory CPU runtime and exact head Grad-CAM."""

import hashlib
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

root = Path(__file__).resolve().parents[1]
source = root / "backend/models/brain_tumor_multiclass.keras"
model = tf.keras.models.load_model(source, compile=False)
base = model.get_layer("mobilenetv2_1.00_224")
conv = next(
    layer
    for layer in reversed(base.layers)
    if isinstance(layer, tf.keras.layers.Conv2D)
)
extractor = tf.keras.Model(base.inputs, [conv.output, base.output])
inputs = tf.keras.Input((224, 224, 3))
activations, features = extractor(inputs, training=False)
head = model.layers[model.layers.index(base) + 1 :]
outputs = features
for layer in head:
    outputs = layer(outputs, training=False)
portable = tf.keras.Model(inputs, [outputs, activations])
converter = tf.lite.TFLiteConverter.from_keras_model(portable)
converted = converter.convert()
(root / "backend/models/brain_tumor_float32.tflite").write_bytes(converted)
bn = base.layers[base.layers.index(conv) + 1]
assert isinstance(bn, tf.keras.layers.BatchNormalization)
relu = base.layers[base.layers.index(conv) + 2]
assert isinstance(relu, tf.keras.layers.ReLU) and relu.max_value == 6
assert all(
    isinstance(
        layer,
        (
            tf.keras.layers.GlobalAveragePooling2D,
            tf.keras.layers.Dense,
            tf.keras.layers.Dropout,
        ),
    )
    for layer in head
)
dense = [layer for layer in head if isinstance(layer, tf.keras.layers.Dense)]
gamma, beta, mean, variance = bn.get_weights()
params = {
    "gamma": gamma,
    "beta": beta,
    "mean": mean,
    "variance": variance,
    "epsilon": np.array(bn.epsilon),
    "source_sha256": np.array(hashlib.sha256(source.read_bytes()).hexdigest()),
}
for i, layer in enumerate(dense):
    params[f"w{i}"] = layer.kernel.numpy()
    params[f"b{i}"] = layer.bias.numpy()
np.savez_compressed(root / "backend/models/gradcam_head.npz", **params)
metadata = {
    "source_sha256": str(params["source_sha256"]),
    "tflite_sha256": hashlib.sha256(converted).hexdigest(),
    "head_sha256": hashlib.sha256(
        (root / "backend/models/gradcam_head.npz").read_bytes()
    ).hexdigest(),
    "quantization": "none; float32",
    "conv_layer": conv.name,
    "gradcam": "Analytic chain rule through BatchNorm, ReLU6, GAP and original Dense/ReLU head",
}
(root / "backend/models/lite_metadata.json").write_text(json.dumps(metadata, indent=2))
print(metadata)
