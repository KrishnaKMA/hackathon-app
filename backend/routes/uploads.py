"""Shared bounded upload validation and model readiness checks."""

import asyncio
import io
import warnings

from fastapi import HTTPException
from PIL import Image, UnidentifiedImageError

from ..config import MAX_IMAGE_PIXELS, MAX_UPLOAD_BYTES


def require_model(request):
    if request.app.state.model is None:
        raise HTTPException(503, "ML model is not ready")
    return request.app.state.model


async def read_image(file):
    contents = await file.read(MAX_UPLOAD_BYTES + 1)
    if not contents:
        raise HTTPException(400, "Empty image")
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Image too large")
    await asyncio.to_thread(validate_image, contents)
    return contents


def validate_image(contents: bytes) -> None:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(contents)) as image:
                if image.width * image.height > MAX_IMAGE_PIXELS:
                    raise HTTPException(400, "Image exceeds decoded pixel limit")
                if image.format not in {"JPEG", "PNG", "WEBP", "BMP"}:
                    raise HTTPException(415, "Use a JPEG, PNG, WebP or BMP image")
                image.verify()
    except (
        UnidentifiedImageError,
        OSError,
        SyntaxError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ):
        raise HTTPException(400, "Invalid or excessively large image") from None
