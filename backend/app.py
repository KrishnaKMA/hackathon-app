"""FastAPI application for brain tumor detection."""

import logging
import os
import numpy as np

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from config import CLASS_NAMES, CLASS_INFO, NUM_CLASSES
from model_loader import load_model
from utils import preprocess_image, generate_heatmap_image

# --------------------------------------------------
# App + logging (Azure-safe)
# --------------------------------------------------
app = FastAPI(title="Brain Tumor Detection API")
logging.basicConfig(level=logging.INFO)

# --------------------------------------------------
# CORS (frontend-safe)
# --------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------
# Load model on startup (Azure-safe)
# --------------------------------------------------
try:
    model = load_model()
    logging.info("Model loaded successfully")
except Exception:
    logging.exception("Model failed to load")
    model = None

# --------------------------------------------------
# Root route (CRITICAL for Azure App Service)
# --------------------------------------------------
@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "brain-tumor-backend",
        "num_classes": NUM_CLASSES,
    }

# --------------------------------------------------
# Health check (used for Azure debugging)
# --------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "num_classes": NUM_CLASSES,
        "classes": CLASS_NAMES,
        "model_loaded": model is not None,
        "environment": os.getenv("WEBSITE_SITE_NAME", "local"),
    }

# --------------------------------------------------
# Predict endpoint
# --------------------------------------------------
@app.post("/predict")
@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    contents = await file.read()

    # Preprocess image
    x = preprocess_image(contents)

    # Predict (4-class probabilities)
    probs = model.predict(x, verbose=0)[0]
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

    return {
        "predicted_class": pred_class,
        "predicted_display": CLASS_INFO[pred_class]["display"],
        "confidence_pct": round(confidence * 100, 2),
        "confidence": round(confidence, 4),
        "all_predictions": predictions,
        "is_tumor": pred_class != "notumor",
        "message": (
            f"Tumor detected: {CLASS_INFO[pred_class]['display']} "
            f"(confidence: {round(confidence * 100, 2)}%)"
            if pred_class != "notumor"
            else f"No tumor detected (confidence: {round(confidence * 100, 2)}%)"
        ),
    }

# --------------------------------------------------
# Heatmap endpoint
# --------------------------------------------------
@app.post("/heatmap")
@app.post("/api/heatmap")
async def get_heatmap(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    contents = await file.read()
    heatmap_b64 = generate_heatmap_image(model, contents)
    return {"heatmap": heatmap_b64}
