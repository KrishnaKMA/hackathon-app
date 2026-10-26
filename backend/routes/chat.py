from fastapi import APIRouter

from ..schemas.chat import ChatRequest, ChatResponse
from ..services.llm_service import answer_question

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    return await answer_question(request.message, request.context)
