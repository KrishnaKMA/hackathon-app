import os

from fastapi import APIRouter, HTTPException, Request

from ..config import CLASS_NAMES, NUM_CLASSES

router = APIRouter()


@router.get("/health")
def health(request: Request):
    return {
        "status": "ok",
        "model_status": request.app.state.model_status,
        "model_loaded": request.app.state.model is not None,
        "initialization_seconds": request.app.state.model_load_seconds,
        "model_error": "Model failed to load"
        if request.app.state.model_status == "error"
        else None,
        "num_classes": NUM_CLASSES,
        "classes": CLASS_NAMES,
    }


@router.get("/ready")
def ready(request: Request):
    if request.app.state.model is None:
        raise HTTPException(503, "ML model is not ready")
    return {
        "status": "ready",
        "revision": os.getenv("RENDER_GIT_COMMIT"),
        "model_backend": getattr(request.app.state.model, "backend", "keras"),
        "runtime_sha256": getattr(request.app.state.model, "runtime_sha256", None),
        "model_sha256": getattr(request.app.state.model, "_neurodetect_sha256", None),
    }
