"""流水线编排 - 实现分布式 RAG 检索流水线。

将 rag_langgraph.graphs.retrieval_graph 中的逻辑重新实现为
一系列调用：Milvus（本地）+ 推理 Worker（GPU，通过 HTTP）。
"""

import json
import logging

from server.config import (
    COLLECTION_NAME, EMBEDDING_DIM, DEFAULT_HYBRID_ALPHA, DEFAULT_SEARCH_K,
)
from server.services import inference_client
from server.services.bm25_search import get_bm25_index

logger = logging.getLogger(__name__)


def _normalize_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []
    min_s, max_s = min(scores), max(scores)
    if max_s == min_s:
        return [1.0] * len(scores)
    return [(s - min_s) / (max_s - min_s) for s in scores]


def _hybrid_fuse(
    dense_results: list[dict],
    sparse_results: list[dict],
    alpha: float = 0.3,
    k: int = 100,
) -> list[dict]:
    dense_scores = _normalize_scores([r["score"] for r in dense_results])
    sparse_scores = _normalize_scores([r["score"] for r in sparse_results])

    score_map: dict[str, dict] = {}
    for i, r in enumerate(dense_results):
        key = r["content"][:200]
        score_map[key] = score_map.get(key, {"content": r["content"], "dense": 0, "sparse": 0, "metadata": r.get("metadata", {})})
        score_map[key]["dense"] = dense_scores[i]

    for i, r in enumerate(sparse_results):
        key = r["content"][:200]
        if key not in score_map:
            score_map[key] = {"content": r["content"], "dense": 0, "sparse": 0, "metadata": r.get("metadata", {})}
        score_map[key]["sparse"] = sparse_scores[i]

    fused = []
    for key, item in score_map.items():
        combined = alpha * item["sparse"] + (1 - alpha) * item["dense"]
        fused.append({"content": item["content"], "score": combined, "metadata": item["metadata"]})

    fused.sort(key=lambda x: x["score"], reverse=True)
    return fused[:k]


def _repack(documents: list[dict], method: str = "sides") -> str:
    """将文档重新打包为单个上下文字符串。"""
    contents = [d["content"] for d in documents]
    if method == "sides":
        result = []
        left, right = 0, len(contents) - 1
        while left <= right:
            result.append(contents[left])
            if left != right:
                result.append(contents[right])
            left += 1
            right -= 1
        return "\n".join(result)
    elif method == "compact_reverse":
        return "\n".join(reversed(contents))
    else:  # compact
        return "\n".join(contents)


def run_query_pipeline(
    query: str,
    search_method: str = "hyde_with_hybrid",
    rerank_model: str = "monot5",
    top_k: int = 10,
    repack_method: str = "sides",
    compression_method: str = "recomp_extractive",
    compression_ratio: float = 0.6,
    hybrid_alpha: float = 0.3,
    search_k: int = 100,
    collection_name: str = COLLECTION_NAME,
) -> dict:
    """运行完整的 RAG 查询流水线，分布在 RAG 服务端和 GPU 之间。"""

    # --- 第 1 步：分类 ---
    label = inference_client.classify(query)

    if label == 0:
        return {
            "answer": "No retrieval needed for this query.",
            "retrieved_documents": [],
            "reranked_documents": [],
            "hyde_document": None,
            "classification_label": label,
        }

    # --- 第 2 步：HyDE（可选）---
    hyde_document = None
    effective_query = query
    if search_method in ("hyde", "hyde_with_hybrid"):
        hyde_document = inference_client.hyde(query)
        effective_query = hyde_document

    # --- 第 3 步：稠密检索（Milvus，本地）---
    dense_results = []
    if search_method in ("original", "hyde", "hybrid", "hyde_with_hybrid"):
        query_embedding = inference_client.embed([effective_query])[0]
        from rag_langgraph.indexing.vectorstore import MilvusVectorStore
        store = MilvusVectorStore(collection_name=collection_name, dim=EMBEDDING_DIM)
        try:
            dense_results = store.search(query_embedding, top_k=search_k)
        except Exception as e:
            logger.warning(f"Milvus search failed: {e}, returning empty results")

    # --- 第 4 步：稀疏检索（BM25，本地）---
    sparse_results = []
    if search_method in ("hybrid", "hyde_with_hybrid", "bm25"):
        bm25 = get_bm25_index(collection_name)
        sparse_results = bm25.search(query, top_k=search_k)

    # --- 第 5 步：合并结果 ---
    if search_method == "bm25":
        final_results = sparse_results
    elif search_method in ("hybrid", "hyde_with_hybrid"):
        final_results = _hybrid_fuse(dense_results, sparse_results, alpha=hybrid_alpha, k=search_k)
    else:
        final_results = dense_results

    # --- 第 6 步：重排序（GPU）---
    reranked = inference_client.rerank(query, final_results, model=rerank_model, top_k=top_k)

    # --- 第 7 步：重打包（本地）---
    context = _repack(reranked, method=repack_method)

    # --- 第 8 步：压缩（GPU）---
    compressed = inference_client.compress(query, context, method=compression_method, ratio=compression_ratio)

    # --- 第 9 步：生成（GPU）---
    answer = inference_client.generate(query, context=compressed)

    return {
        "answer": answer,
        "retrieved_documents": final_results[:top_k],
        "reranked_documents": reranked,
        "hyde_document": hyde_document,
        "classification_label": label,
    }


def run_indexing_pipeline(
    data_path: str,
    collection_name: str = COLLECTION_NAME,
    chunk_size: int = 512,
    chunk_overlap: int = 20,
) -> dict:
    """运行索引流水线：加载 -> 切分 -> 嵌入（GPU）-> 存储（Milvus）。"""

    from rag_langgraph.indexing.loader import load_from_directory, load_from_json
    from rag_langgraph.indexing.splitter import split_documents

    # 加载文档
    import os
    if os.path.isdir(data_path):
        documents = load_from_directory(data_path)
    else:
        documents = load_from_json(data_path)

    if not documents:
        return {"document_count": 0, "message": "No documents found"}

    # 切分
    chunks = split_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = [c["content"] for c in chunks]

    # 通过 GPU 生成嵌入
    embeddings = inference_client.embed(texts)

    # 存入 Milvus
    from rag_langgraph.indexing.vectorstore import MilvusVectorStore
    from server.config import EMBEDDING_DIM
    store = MilvusVectorStore(collection_name=collection_name, dim=EMBEDDING_DIM)
    metadata_strs = [json.dumps(c.get("metadata", {}), ensure_ascii=False) for c in chunks]
    store.insert(embeddings, texts, metadata_strs)
    store.flush()

    # 在本地构建 BM25 索引
    from server.services.bm25_search import get_bm25_index
    bm25 = get_bm25_index(collection_name)
    bm25.build(texts)
    bm25.save(collection_name)

    stats = store.get_stats()
    return {
        "document_count": stats.get("row_count", len(texts)),
        "message": f"Indexed {len(texts)} chunks into {collection_name}",
    }
