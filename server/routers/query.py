"""查询路由 - 处理 RAG 查询请求。"""

from fastapi import APIRouter
from pydantic import BaseModel

from server.services.pipeline import run_query_pipeline

router = APIRouter()


class QueryRequest(BaseModel):
    query: str
    collection: str | None = None
    search_method: str = "hyde_with_hybrid"
    rerank_model: str = "monot5"
    top_k: int = 10
    repack_method: str = "sides"
    compression_method: str = "recomp_extractive"
    compression_ratio: float = 0.6
    hybrid_alpha: float = 0.3
    search_k: int = 100


class DocumentResult(BaseModel):
    content: str
    score: float
    metadata: dict = {}


class QueryResponse(BaseModel):
    retrieved_documents: list[DocumentResult] = []
    reranked_documents: list[DocumentResult] = []
    repacked_context: str = ""
    compressed_context: str = ""
    hyde_document: str | None = None
    classification_label: int | None = None


@router.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    result = run_query_pipeline(
        query=req.query,
        collection_name=req.collection or None,
        search_method=req.search_method,
        rerank_model=req.rerank_model,
        top_k=req.top_k,
        repack_method=req.repack_method,
        compression_method=req.compression_method,
        compression_ratio=req.compression_ratio,
        hybrid_alpha=req.hybrid_alpha,
        search_k=req.search_k,
    )
    return QueryResponse(**result)
