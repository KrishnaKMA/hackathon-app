from fastapi import APIRouter, File, Request, UploadFile

from ..schemas.prediction import HeatmapResponse
from ..services.heatmap_service import generate_heatmap_image
from ..services.runtime import run_model_task
from .uploads import read_image, require_model

router = APIRouter()


@router.post("/heatmap", response_model=HeatmapResponse)
async def heatmap(request: Request, file: UploadFile = File(...)):
    model = require_model(request)
    contents = await read_image(file)
    result = await run_model_task(
        request.app.state, generate_heatmap_image, model, contents
    )
    return {"heatmap": result}
