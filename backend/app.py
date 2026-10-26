"""FastAPI entry point. Run from the repository root: uvicorn backend.app:app."""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from .config import CORS_ORIGINS, LOG_LEVEL
from .model_loader import load_model
from .routes import chat, health, heatmap, predict

logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger("neurodetect")


async def load_model_in_background(app):
    start = time.perf_counter()
    try:
        app.state.model = await asyncio.to_thread(load_model)
        app.state.model_load_seconds = time.perf_counter() - start
        app.state.model_status = "ready"
        logger.info("ML model loaded successfully")
    except Exception:
        app.state.model_status = "error"
        logger.exception("ML model failed to load")


@asynccontextmanager
async def lifespan(app):
    app.state.model = None
    app.state.model_status = "loading"
    app.state.model_load_seconds = None
    app.state.inference_lock = asyncio.Lock()
    task = asyncio.create_task(
        load_model_in_background(app), name="model-initialization"
    )
    app.state.loading_task = task
    try:
        yield
    finally:
        # Worker threads cannot be force-cancelled; finish initialization on shutdown.
        await task


app = FastAPI(title="NeuroDetect", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)
for module in (health, predict, heatmap, chat):
    app.include_router(module.router, prefix="/api")
    app.include_router(module.router, include_in_schema=False)


@app.get("/")
def root():
    return {"status": "ok"}


@app.exception_handler(Exception)
async def unhandled_exception_handler(request, exc):
    logger.error("Unhandled server error", exc_info=(type(exc), exc, exc.__traceback__))
    return JSONResponse(status_code=500, content={"error": "Internal server error"})


# The frontend remains separate source code; this optional mount provides a free
# same-service deployment when no separately hosted frontend is configured.
frontend_path = Path(__file__).resolve().parents[1] / "frontend"
if frontend_path.is_dir():
    app.mount("/ui", StaticFiles(directory=frontend_path, html=True), name="frontend")
