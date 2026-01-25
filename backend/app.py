"""FastAPI application for brain tumor detection."""

import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import httpx

from .config import CLASS_NAMES, CLASS_INFO, NUM_CLASSES, GEMINI_API_KEY as CONFIG_GEMINI_API_KEY
from .model_loader import load_model
from .utils import preprocess_image, generate_heatmap_image

app = FastAPI()


class ChatRequest(BaseModel):
    message: str
    context: dict | None = None

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

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", CONFIG_GEMINI_API_KEY)
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")


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


@app.post("/chat")
@app.post("/api/chat")
async def chat_with_assistant(request: ChatRequest):
    """
    Chat with AI assistant about brain tumor detection results.

    Args:
        request: ChatRequest with message and optional prediction context

    Returns:
        Dictionary with assistant response
    """
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not set on server.")

    system_prompt = (
        "You are a helpful medical AI assistant for NeuroDetect, a brain tumor detection system. "
        "You help users understand their MRI scan analysis results. You can explain: "
        "- What different tumor types mean (glioma, meningioma, pituitary) "
        "- How to interpret confidence scores "
        "- General information about brain tumors "
        "- What the Grad-CAM heatmap shows "
        "IMPORTANT: Always remind users that this is for educational purposes only and they should consult "
        "a qualified medical professional for actual diagnosis and treatment decisions. "
        "Be concise, empathetic, and helpful. If prediction context is provided, reference it in your response."
    )

    user_message = request.message
    if request.context:
        context_str = f"\n\nCurrent scan results: {request.context}"
        user_message = f"{user_message}{context_str}"

    prompt = f"{system_prompt}\n\nUser: {user_message}"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
    headers = {"x-goog-api-key": GEMINI_API_KEY, "Content-Type": "application/json"}
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 1024},
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(url, headers=headers, json=payload, timeout=30.0)
        if response.status_code != 200:
            raise HTTPException(status_code=502, detail=response.text)

        data = response.json()
        text = (
            data.get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [{}])[0]
            .get("text", "")
        )
        if not text:
            raise HTTPException(status_code=502, detail="Empty response from Gemini.")

        reply = text.strip()
        return {"response": reply, "reply": reply, "error": None}
