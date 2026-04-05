"""重排序推理端点。

封装 rag_langgraph.models.rerankers 的调度逻辑。
"""

from fastapi import APIRouter
from pydantic import BaseModel

from rag_langgraph.models.rerankers import get_reranker

router = APIRouter()


class Document(BaseModel):
    content: str
    score: float = 0.0
    metadata: dict = {}


class RerankRequest(BaseModel):
    query: str
    documents: list[Document]
    model: str = "monot5"
    top_k: int = 10


class RerankResponse(BaseModel):
    documents: list[Document]


@router.post("/rerank", response_model=RerankResponse)
def rerank(req: RerankRequest):
    reranker = get_reranker(model_name=req.model)
    docs = [d.model_dump() for d in req.documents]
    results = reranker.rerank(req.query, docs, top_k=req.top_k)
    return RerankResponse(
        documents=[Document(**d) for d in results]
    )
