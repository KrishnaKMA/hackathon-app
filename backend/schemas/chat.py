from typing import Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000, pattern=r"\S")
    context: dict | None = None


class ChatResponse(BaseModel):
    response: str
    status: Literal["ok", "degraded", "error"]
    error: str | None = None
