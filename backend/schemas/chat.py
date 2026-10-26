from typing import Literal

from pydantic import BaseModel, Field


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=1000, pattern=r"\S")


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000, pattern=r"\S")
    context: dict | None = None
    history: list[ChatTurn] = Field(default_factory=list, max_length=6)


class ChatResponse(BaseModel):
    response: str
    status: Literal["ok", "degraded", "error"]
    error: str | None = None
