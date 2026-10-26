"""Image preprocessing and classification; executed off the event loop."""

import time

import numpy as np

from ..config import CLASS_INFO, CLASS_NAMES, INFERENCE_MODE
from ..utils import preprocess_image


def predict_scan(model, contents, timings=None):
    timings = timings if timings is not None else {}
    start = time.perf_counter()
    # Preprocess image
    x = preprocess_image(contents)

    timings["preprocess"] = (time.perf_counter() - start) * 1000
    start = time.perf_counter()

    # Predict (4-class probabilities)
    if INFERENCE_MODE == "predict":
        probs = np.asarray(model.predict(x, verbose=0))[0]
    elif INFERENCE_MODE == "compiled" and hasattr(model, "_neurodetect_infer"):
        probs = np.asarray(model._neurodetect_infer(x))[0]
    else:
        probs = np.asarray(model(x, training=False))[0]
    timings["inference"] = (time.perf_counter() - start) * 1000
    start = time.perf_counter()
    pred_idx = int(np.argmax(probs))
    pred_class = CLASS_NAMES[pred_idx]
    confidence = float(probs[pred_idx])

    # Build response with all class probabilities
    predictions = {}
    for i, class_name in enumerate(CLASS_NAMES):
        class_prob = float(probs[i])
        predictions[class_name] = {
            "probability": round(class_prob, 4),
            "confidence_pct": round(class_prob * 100, 2),
            "display": CLASS_INFO[class_name]["display"],
            "type": CLASS_INFO[class_name]["type"],
        }

    result = {
        "predicted_class": pred_class,
        "predicted_display": CLASS_INFO[pred_class]["display"],
        "confidence_pct": round(confidence * 100, 2),
        "confidence": round(confidence, 4),
        "all_predictions": predictions,
        "is_tumor": pred_class != "notumor",
        "message": (
            f"Model predicts: {CLASS_INFO[pred_class]['display']} "
            f"(confidence: {round(confidence * 100, 2)}%)"
            if pred_class != "notumor"
            else f"Model predicts no tumor (confidence: {round(confidence * 100, 2)}%)"
        ),
    }

    timings["response"] = (time.perf_counter() - start) * 1000
    return result
