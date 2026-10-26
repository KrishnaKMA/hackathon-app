from typing import Literal

from pydantic import BaseModel, Field

ClassName = Literal["glioma", "meningioma", "notumor", "pituitary"]


class ClassProbability(BaseModel):
    probability: float = Field(ge=0, le=1)
    confidence_pct: float = Field(ge=0, le=100)
    display: str
    type: str


class PredictionResponse(BaseModel):
    predicted_class: ClassName
    predicted_display: str
    confidence: float = Field(ge=0, le=1)
    confidence_pct: float = Field(ge=0, le=100)
    all_predictions: dict[ClassName, ClassProbability]
    is_tumor: bool
    message: str


class HeatmapResponse(BaseModel):
    heatmap: str
