"""向量嵌入推理端点。

封装 rag_langgraph.indexing.embedding.get_embedder。
"""

from fastapi import APIRouter
from pydantic import BaseModel

from rag_langgraph.indexing.embedding import get_embedder
from inference.config import EMBEDDING_MODEL

router = APIRouter()


class EmbedRequest(BaseModel):
    texts: list[str]
    model_name: str = EMBEDDING_MODEL


class EmbedResponse(BaseModel):
    embeddings: list[list[float]]
    dim: int


@router.post("/embed", response_model=EmbedResponse)
def embed(req: EmbedRequest):
    embedder = get_embedder(model_name=req.model_name)
    embeddings = embedder.embed_batch(req.texts)
    return EmbedResponse(embeddings=embeddings, dim=len(embeddings[0]) if embeddings else 0)
