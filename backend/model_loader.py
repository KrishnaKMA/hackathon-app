"""Initialize and warm up the original trained model once per process."""

import hashlib
import logging
import time

from .config import (
    IMG_SIZE,
    INFERENCE_MODE,
    MODEL_BACKEND,
    MODEL_PATH,
    TF_INTER_THREADS,
    TF_INTRA_THREADS,
)

logger = logging.getLogger("neurodetect")


def load_model():
    start = time.perf_counter()
    import numpy as np

    if MODEL_BACKEND == "tflite":
        from .services.lite_runtime import LiteModel

        model = LiteModel()
        sample = np.zeros((1, *IMG_SIZE, 3), dtype=np.float32)
        for _ in range(3):
            output = model.predict(sample)
        if output.shape != (1, 4) or not np.isfinite(output).all():
            raise ValueError("Unexpected Lite model output during warm-up")
        logger.info(
            "Lite model initialized and warmed in %.2f seconds",
            time.perf_counter() - start,
        )
        return model
    if MODEL_BACKEND != "keras":
        raise ValueError("MODEL_BACKEND must be keras or tflite")
    import tensorflow as tf

    if TF_INTRA_THREADS:
        tf.config.threading.set_intra_op_parallelism_threads(TF_INTRA_THREADS)
    if TF_INTER_THREADS:
        tf.config.threading.set_inter_op_parallelism_threads(TF_INTER_THREADS)
    if not MODEL_PATH.is_file():
        raise FileNotFoundError("Configured model file does not exist")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    if INFERENCE_MODE not in {"predict", "direct", "compiled"}:
        raise ValueError("INFERENCE_MODE must be compiled, direct, or predict")
    if INFERENCE_MODE == "compiled":
        model._neurodetect_infer = tf.function(
            lambda x: model(x, training=False),
            input_signature=[tf.TensorSpec((1, *IMG_SIZE, 3), tf.float32)],
        )
    model._neurodetect_sha256 = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()
    sample = np.zeros((1, *IMG_SIZE, 3), dtype=np.float32)
    for _ in range(3):
        if INFERENCE_MODE == "predict":
            output = model.predict(sample, verbose=0)
        elif INFERENCE_MODE == "compiled":
            output = model._neurodetect_infer(sample).numpy()
        else:
            output = model(sample, training=False).numpy()
    if output.shape != (1, 4) or not np.all(np.isfinite(output)):
        raise ValueError("Unexpected model output during warm-up")
    logger.info(
        "Model initialized and warmed in %.2f seconds", time.perf_counter() - start
    )
    return model
