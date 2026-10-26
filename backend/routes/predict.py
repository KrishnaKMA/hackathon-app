import time

from fastapi import APIRouter, File, Request, Response, UploadFile

from ..schemas.prediction import PredictionResponse
from ..services.ml_service import predict_scan
from ..services.runtime import run_model_task
from .uploads import read_image, require_model

router = APIRouter()


@router.post("/predict", response_model=PredictionResponse)
async def predict(request: Request, response: Response, file: UploadFile = File(...)):
    start = time.perf_counter()
    timings = {}
    model = require_model(request)
    contents = await read_image(file)
    timings["upload_validate"] = (time.perf_counter() - start) * 1000
    result = await run_model_task(
        request.app.state, predict_scan, model, contents, timings, timings=timings
    )
    timings["total"] = (time.perf_counter() - start) * 1000
    response.headers["Server-Timing"] = ", ".join(
        f"{key};dur={value:.3f}" for key, value in timings.items()
    )
    return result
