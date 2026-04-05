"""
重排序节点。

使用多种重排序模型按相关性对检索到的文档重新排序：
- MonoT5（推荐）：基于 T5 的序列到序列相关性评分
- BGE: 交叉编码器评分（BAAI/bge-reranker-v2-m3）
- RankLLaMA: 基于 LLaMA 并使用 LoRA 微调
- TILDE: 查询扩展 + BM25 风格评分
"""

import logging

from rag_langgraph.state import RAGState, Document

logger = logging.getLogger(__name__)


def reranking_node(state: RAGState) -> dict:
    """
    使用配置的重排序模型对检索到的文档进行重排序。

    Args:
        state: 包含 retrieved_documents 的当前 RAG 流水线状态。

    Returns:
        更新后的状态，包含 reranked_documents。
    """
    from rag_langgraph.models.rerankers import get_reranker

    query = state["query"]
    documents = state["retrieved_documents"]
    rerank_model = state.get("rerank_model", "monot5")
    top_k = state.get("top_k", 10)

    if not documents:
        return {"reranked_documents": []}

    reranker = get_reranker(rerank_model)

    # 提取内容用于重排序
    doc_contents = [doc["content"] for doc in documents]
    scored_docs = reranker.rerank(query, doc_contents)

    # 按分数降序排序并取 top_k
    scored_docs.sort(key=lambda x: x["score"], reverse=True)
    scored_docs = scored_docs[:top_k]

    # 映射回 Document 格式
    reranked_documents = [
        Document(
            content=scored_doc["content"],
            score=scored_doc["score"],
            metadata=scored_doc.get("metadata", {}),
        )
        for scored_doc in scored_docs
    ]

    return {"reranked_documents": reranked_documents}
