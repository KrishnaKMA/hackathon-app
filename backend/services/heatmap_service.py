"""Grad-CAM over MobileNetV2's final convolution, reusing trained weights."""

import base64
import io
import weakref

import numpy as np
from PIL import Image

from ..utils import preprocess_image

_extractors = weakref.WeakKeyDictionary()


def gradcam_values(model, contents: bytes):
    if getattr(model, "backend", None) == "tflite":
        return model.gradcam(preprocess_image(contents))
    import tensorflow as tf

    base = model.get_layer("mobilenetv2_1.00_224")
    if model not in _extractors:
        layer = next(
            layer
            for layer in reversed(base.layers)
            if isinstance(layer, tf.keras.layers.Conv2D)
        )
        _extractors[model] = tf.keras.Model(base.inputs, [layer.output, base.output])
    extractor = _extractors[model]
    x = tf.convert_to_tensor(preprocess_image(contents))
    head = model.layers[model.layers.index(base) + 1 :]
    with tf.GradientTape() as tape:
        # Backbone is frozen; explicitly track the input to record its operations.
        tape.watch(x)
        activations, features = extractor(x, training=False)
        scores = features
        for layer in head[:-1]:
            scores = layer(scores, training=False)
        final = head[-1]
        # Pre-softmax class score avoids saturated softmax gradients, without mutation.
        logits = tf.linalg.matmul(scores, final.kernel) + final.bias
        index = tf.argmax(logits[0])
        score = logits[0, index]
    gradients = tape.gradient(score, activations)
    if gradients is None:
        raise ValueError("Grad-CAM gradient path is disconnected")
    weights = tf.reduce_mean(gradients, axis=(0, 1, 2))
    cam = tf.nn.relu(tf.reduce_sum(activations[0] * weights, axis=-1)).numpy()
    if not np.all(np.isfinite(cam)):
        raise ValueError("Non-finite Grad-CAM values")
    maximum = float(cam.max())
    return cam / maximum if maximum > 0 else np.zeros_like(cam)


def generate_heatmap_image(model, contents: bytes) -> str:
    cam = gradcam_values(model, contents)
    with Image.open(io.BytesIO(contents)) as image:
        original = np.asarray(image.convert("RGB"), dtype=np.float32)
        resized = np.asarray(
            Image.fromarray(cam).resize(image.size, Image.Resampling.BILINEAR)
        )
    # Display only stronger relative influence; retain the unmodified Grad-CAM.
    # This is a visualization transfer function, not a tumor segmentation mask.
    strength = np.clip((resized - 0.35) / 0.65, 0, 1)
    alpha = (0.60 * strength**1.5)[..., None]
    # Blend in-place to bound memory even at the configured four-million-pixel limit.
    original *= 1 - alpha
    original[..., 0] += 255 * alpha[..., 0]
    np.clip(original, 0, 255, out=original)
    overlay = original.astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(overlay).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode(
        "ascii"
    )
