"""Pipeline orchestration for the distributed retrieval-first RAG deployment."""

import json
import logging

from rag_langgraph.nodes.repacking import repacking_node
from server.config import COLLECTION_NAME, EMBEDDING_DIM
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
    for i, result in enumerate(dense_results):
        key = result["content"][:200]
        score_map[key] = score_map.get(
            key,
            {
                "content": result["content"],
                "dense": 0,
                "sparse": 0,
                "metadata": result.get("metadata", {}),
            },
        )
        score_map[key]["dense"] = dense_scores[i]

    for i, result in enumerate(sparse_results):
        key = result["content"][:200]
        if key not in score_map:
            score_map[key] = {
                "content": result["content"],
                "dense": 0,
                "sparse": 0,
                "metadata": result.get("metadata", {}),
            }
        score_map[key]["sparse"] = sparse_scores[i]

    fused = []
    for item in score_map.values():
        combined = alpha * item["sparse"] + (1 - alpha) * item["dense"]
        fused.append(
            {
                "content": item["content"],
                "score": combined,
                "metadata": item["metadata"],
            }
        )

    fused.sort(key=lambda x: x["score"], reverse=True)
    return fused[:k]


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
    collection_name: str | None = COLLECTION_NAME,
) -> dict:
    """Run the retrieval-first query pipeline and return retrieval context outputs."""
    collection_name = collection_name or COLLECTION_NAME

    # Classification is intentionally bypassed in this deployment. All queries go through retrieval.
    classification_label = 1

    hyde_document = None
    effective_query = query
    if search_method in ("hyde", "hyde_with_hybrid"):
        hyde_document = inference_client.hyde(query)
        effective_query = hyde_document

    dense_results = []
    if search_method in ("original", "hyde", "hybrid", "hyde_with_hybrid"):
        query_embedding = inference_client.embed([effective_query])[0]
        from rag_langgraph.indexing.vectorstore import MilvusVectorStore

        store = MilvusVectorStore(collection_name=collection_name, dim=EMBEDDING_DIM)
        try:
            dense_results = store.search(query_embedding, top_k=search_k)
        except Exception as exc:
            logger.warning("Milvus search failed: %s", exc)

    sparse_results = []
    if search_method in ("hybrid", "hyde_with_hybrid", "bm25"):
        bm25 = get_bm25_index(collection_name)
        sparse_results = bm25.search(query, top_k=search_k)

    if search_method == "bm25":
        final_results = sparse_results
    elif search_method in ("hybrid", "hyde_with_hybrid"):
        final_results = _hybrid_fuse(dense_results, sparse_results, alpha=hybrid_alpha, k=search_k)
    else:
        final_results = dense_results

    reranked_documents = inference_client.rerank(query, final_results, model=rerank_model, top_k=top_k)
    repacked_context = repacking_node(
        {
            "reranked_documents": reranked_documents,
            "repack_method": repack_method,
        }
    )["repacked_context"]
    compressed_context = ""
    if repacked_context:
        compressed_context = inference_client.compress(
            query,
            repacked_context,
            method=compression_method,
            ratio=compression_ratio,
        )

    return {
        "retrieved_documents": final_results[:top_k],
        "reranked_documents": reranked_documents,
        "repacked_context": repacked_context,
        "compressed_context": compressed_context,
        "hyde_document": hyde_document,
        "classification_label": classification_label,
    }


def run_indexing_pipeline(
    data_path: str,
    collection_name: str = COLLECTION_NAME,
    chunk_size: int = 512,
    chunk_overlap: int = 20,
) -> dict:
    """Run indexing: load -> split -> embed -> store."""
    from rag_langgraph.indexing.loader import load_from_directory, load_from_json
    from rag_langgraph.indexing.splitter import split_documents

    import os

    if os.path.isdir(data_path):
        documents = load_from_directory(data_path)
    else:
        documents = load_from_json(data_path)

    if not documents:
        return {"document_count": 0, "message": "No documents found"}

    chunks = split_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = [chunk["content"] for chunk in chunks]

    embeddings = inference_client.embed(texts)

    from rag_langgraph.indexing.vectorstore import MilvusVectorStore

    store = MilvusVectorStore(collection_name=collection_name, dim=EMBEDDING_DIM)
    metadata_strs = [json.dumps(chunk.get("metadata", {}), ensure_ascii=False) for chunk in chunks]
    store.insert(embeddings, texts, metadata_strs)
    store.flush()

    bm25 = get_bm25_index(collection_name)
    bm25.build(texts)
    bm25.save(collection_name)

    stats = store.get_stats()
    return {
        "document_count": stats.get("row_count", len(texts)),
        "message": f"Indexed {len(texts)} chunks into {collection_name}",
    }
