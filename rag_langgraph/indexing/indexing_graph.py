"""
索引流水线 - LangGraph StateGraph。

构建知识库索引流水线：
  文档 -> 分块 -> 元数据提取 -> 嵌入 -> 向量存储
"""

import json
import logging
from typing import Optional

from langgraph.graph import END, StateGraph

from rag_langgraph.state import IndexingState
from rag_langgraph.config import IndexingConfig

logger = logging.getLogger(__name__)


def load_node(state: IndexingState) -> dict:
    """从指定路径加载文档。"""
    from rag_langgraph.indexing.loader import load_from_directory, load_from_json
    import os

    data_path = state["data_path"]
    if os.path.isdir(data_path):
        documents = load_from_directory(data_path)
    elif data_path.endswith(".json"):
        documents = load_from_json(data_path)
    else:
        documents = []

    return {"raw_documents": documents}


def split_node(state: IndexingState) -> dict:
    """将文档分割为块。"""
    from rag_langgraph.indexing.splitter import split_documents

    documents = state["raw_documents"]
    chunk_size = state.get("chunk_size", 512)
    chunk_overlap = state.get("chunk_overlap", 20)

    chunks = split_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return {"chunks": [c["content"] for c in chunks], "_chunk_dicts": chunks}


def metadata_node(state: IndexingState) -> dict:
    """从块中提取元数据。"""
    from rag_langgraph.indexing.metadata import extract_metadata

    # 使用 split_node 中存储的块字典
    chunk_dicts = state.get("_chunk_dicts", [])
    if not chunk_dicts:
        chunk_dicts = [{"content": c, "metadata": {}} for c in state.get("chunks", [])]

    enriched = extract_metadata(chunk_dicts)
    return {"extracted_metadata": [c["metadata"] for c in enriched], "_enriched_chunks": enriched}


def embed_node(state: IndexingState) -> dict:
    """为所有块生成嵌入向量。"""
    from rag_langgraph.indexing.embedding import get_embedder

    model_name = state.get("embedding_model", "BAAI/bge-base-en-v1.5")
    chunks = state.get("chunks", [])

    embedder = get_embedder(model_name=model_name)
    embeddings = embedder.embed_batch(chunks)

    return {"embeddings": embeddings}


def store_node(state: IndexingState) -> dict:
    """将嵌入向量和文档存储到 Milvus。"""
    from rag_langgraph.indexing.vectorstore import MilvusVectorStore

    collection_name = state.get("collection_name", "rag_collection")
    embeddings = state.get("embeddings", [])
    chunks = state.get("chunks", [])
    metadata = state.get("extracted_metadata", [])

    store = MilvusVectorStore(collection_name=collection_name, dim=state.get("embedding_dim", 768))

    # 将元数据序列化为 JSON 字符串
    metadata_strs = [json.dumps(m, ensure_ascii=False) for m in metadata]

    store.insert(embeddings, chunks, metadata_strs)
    store.flush()

    stats = store.get_stats()
    logger.info(f"Indexing complete. Collection '{collection_name}' has {stats.get('row_count', 0)} documents")

    return {"milvus_id": collection_name}


def build_indexing_graph(config: Optional[IndexingConfig] = None) -> StateGraph:
    """
    将索引流水线构建为 LangGraph StateGraph。

    图结构：
        START -> load -> split -> metadata -> embed -> store -> END

    Args:
        config: 索引配置。

    Returns:
        编译后的 LangGraph StateGraph。
    """
    if config is None:
        config = IndexingConfig()

    graph = StateGraph(IndexingState)

    # 添加节点
    graph.add_node("load", load_node)
    graph.add_node("split", split_node)
    graph.add_node("extract_metadata", metadata_node)
    graph.add_node("embed", embed_node)
    graph.add_node("store", store_node)

    # 添加边
    graph.set_entry_point("load")
    graph.add_edge("load", "split")
    graph.add_edge("split", "extract_metadata")
    graph.add_edge("extract_metadata", "embed")
    graph.add_edge("embed", "store")
    graph.add_edge("store", END)

    compiled = graph.compile()
    logger.info("Indexing graph built and compiled successfully")
    return compiled


def run_indexing(data_path: str, config: Optional[IndexingConfig] = None) -> dict:
    """
    运行完整的索引流水线。

    Args:
        data_path: 文档路径。
        config: 索引配置。

    Returns:
        最终状态字典。
    """
    if config is None:
        config = IndexingConfig(data_path=data_path)

    graph = build_indexing_graph(config)

    initial_state = {
        "data_path": config.data_path or data_path,
        "chunk_size": config.chunk_size,
        "chunk_overlap": config.chunk_overlap,
        "collection_name": config.collection_name,
        "embedding_model": config.embedding_model,
        "embedding_dim": config.embedding_dim,
    }

    result = graph.invoke(initial_state)
    return result
