"""HyDE inference endpoint."""

from fastapi import APIRouter
from pydantic import BaseModel

from inference.config import LLM_MODEL_PATH
from rag_langgraph.models.generator import get_generator

router = APIRouter()


class HyDERequest(BaseModel):
    query: str
    model_path: str = LLM_MODEL_PATH
    max_out_len: int = 100


class HyDEResponse(BaseModel):
    hypothetical_document: str


@router.post("/hyde", response_model=HyDEResponse)
def hyde(req: HyDERequest):
    generator = get_generator(
        model_path=req.model_path,
        max_out_len=req.max_out_len,
    )
    hypothetical = generator.generate_hyde(req.query)
    return HyDEResponse(hypothetical_document=hypothetical)
