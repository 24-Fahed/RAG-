"""文本生成推理端点。

封装 rag_langgraph.models.generator 用于 LLM 答案生成。
同时提供 HyDE（假设性文档嵌入）生成功能。
"""

from fastapi import APIRouter
from pydantic import BaseModel

from rag_langgraph.models.generator import get_generator
from inference.config import LLM_MODEL_PATH

router = APIRouter()


class GenerateRequest(BaseModel):
    query: str
    context: str = ""
    model_path: str = LLM_MODEL_PATH
    max_out_len: int = 50


class GenerateResponse(BaseModel):
    answer: str


class HyDERequest(BaseModel):
    query: str
    model_path: str = LLM_MODEL_PATH
    max_out_len: int = 100


class HyDEResponse(BaseModel):
    hypothetical_document: str


@router.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    generator = get_generator(
        model_path=req.model_path,
        max_out_len=req.max_out_len,
    )
    answer = generator.generate_answer(req.query, req.context)
    return GenerateResponse(answer=answer)


@router.post("/hyde", response_model=HyDEResponse)
def hyde(req: HyDERequest):
    generator = get_generator(
        model_path=req.model_path,
        max_out_len=req.max_out_len,
    )
    hypothetical = generator.generate_hyde(req.query)
    return HyDEResponse(hypothetical_document=hypothetical)
