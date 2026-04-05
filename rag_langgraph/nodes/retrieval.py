"""
检索节点。

支持多种搜索策略：
- original: 直接向量相似度搜索
- hyde: 假设文档嵌入（查询扩展）
- hybrid: 稠密（向量）+ 稀疏（BM25）组合
- hyde_with_hybrid: HyDE + 混合搜索（推荐）
- bm25: 仅 BM25
"""

import logging
from typing import Optional

from rag_langgraph.state import RAGState, Document

logger = logging.getLogger(__name__)


def retrieval_node(state: RAGState) -> dict:
    """
    根据配置的搜索方法执行文档检索。

    支持：original、hyde、hybrid、hyde_with_hybrid、bm25

    Args:
        state: 当前 RAG 流水线状态。

    Returns:
        更新后的状态，包含 retrieved_documents，可选包含 hyde_document。
    """
    from rag_langgraph.models.retriever import get_retriever

    query = state["query"]
    search_method = state.get("search_method", "hyde_with_hybrid")
    search_k = state.get("search_k", 100)
    collection_name = state.get("milvus_collection", "rag_collection")

    retriever = get_retriever(collection_name=collection_name)

    # 步骤 1：HyDE 查询扩展（如需要）
    hyde_document = None
    effective_query = query
    if search_method in ("hyde", "hyde_with_hybrid"):
        hyde_document = retriever.generate_hypothetical_document(query)
        effective_query = hyde_document

    # 步骤 2：稠密检索
    dense_results = []
    if search_method in ("original", "hyde", "hybrid", "hyde_with_hybrid"):
        dense_results = retriever.dense_search(effective_query, top_k=search_k)

    # 步骤 3：稀疏检索（BM25）
    sparse_results = []
    if search_method in ("hybrid", "hyde_with_hybrid", "bm25"):
        sparse_results = retriever.sparse_search(query, top_k=search_k)

    # 步骤 4：根据策略合并结果
    if search_method == "bm25":
        final_results = sparse_results
    elif search_method in ("hybrid", "hyde_with_hybrid"):
        alpha = state.get("hybrid_alpha", 0.3)
        final_results = retriever.hybrid_fuse(dense_results, sparse_results, alpha=alpha, k=search_k)
    else:
        final_results = dense_results

    # 转换为 Document 格式
    documents = [
        Document(content=doc["content"], score=doc["score"], metadata=doc.get("metadata", {}))
        for doc in final_results
    ]

    update = {"retrieved_documents": documents}
    if hyde_document:
        update["hyde_document"] = hyde_document

    return update
