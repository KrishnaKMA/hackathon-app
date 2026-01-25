"""FastAPI application for brain tumor detection."""

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import numpy as np

from config import CLASS_NAMES, CLASS_INFO, NUM_CLASSES
from model_loader import load_model
from utils import preprocess_image, generate_heatmap_image

app = FastAPI()

# allow frontend (we'll tighten later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model on startup
model = load_model()


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok", "num_classes": NUM_CLASSES, "classes": CLASS_NAMES}


@app.post("/predict")
@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict brain tumor class from MRI image (4 classes: glioma, meningioma, notumor, pituitary).
    
    Args:
        file: MRI image file to classify
        
    Returns:
        Dictionary with prediction results for all classes
    """
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
            "type": CLASS_INFO[class_name]["type"]
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
        )
    }


@app.post("/heatmap")
@app.post("/api/heatmap")
async def get_heatmap(file: UploadFile = File(...)):
    """
    Generate a Grad-CAM heatmap showing model attention regions.
    
    Args:
        file: MRI image file to generate heatmap for
        
    Returns:
        Dictionary with base64-encoded heatmap image
    """
    contents = await file.read()
    heatmap_b64 = generate_heatmap_image(model, contents)
    return {"heatmap": heatmap_b64}
