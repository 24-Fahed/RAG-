"""
RAG LangGraph 系统的配置管理。
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RAGConfig:
    """RAG 检索流水线的配置。"""

    # --- 分类 ---
    with_classification: bool = True
    classification_model_path: str = "google-bert/bert-base-multilingual-cased"
    classification_weights_path: str = "bert_best_model.pth"

    # --- 检索 ---
    search_method: str = "hyde_with_hybrid"
    search_k: int = 100
    hybrid_alpha: float = 0.3  # 混合检索中稀疏检索的权重：score = alpha * sparse + dense

    # --- 重排序 ---
    rerank_model: str = "monot5"
    top_k: int = 10

    # --- 重打包 ---
    repack_method: str = "sides"

    # --- 压缩 ---
    compression_method: str = "recomp_extractive"
    compression_ratio: float = 0.6

    # --- 向量存储 ---
    milvus_collection: str = "rag_collection"
    embedding_dim: int = 768
    milvus_host: str = "localhost"
    milvus_port: int = 19530

    # --- 嵌入 ---
    embedding_model: str = "BAAI/bge-base-en-v1.5"

    # --- 大语言模型 ---
    llm_model_path: str = ""
    llm_max_out_len: int = 50

    # --- 重排序模型路径 ---
    monot5_model: str = "castorini/monot5-base-msmarco-10k"
    bge_reranker_model: str = "BAAI/bge-reranker-v2-m3"
    rankllama_model: str = "castorini/rankllama-v1-7b-lora-passage"
    tilde_model: str = "ielab/TILDEv2-TILDE200-exp"

    # --- 压缩模型路径 ---
    recomp_extractive_model: str = "fangyuan/nq_extractive_compressor"
    recomp_abstractive_model: str = "fangyuan/nq_abstractive_compressor"
    llmlingua_model: str = "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"


@dataclass
class IndexingConfig:
    """知识库索引流水线的配置。"""

    data_path: str = ""
    chunk_size: int = 512
    chunk_overlap: int = 20
    collection_name: str = "rag_collection"
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    embedding_dim: int = 768
    milvus_host: str = "localhost"
    milvus_port: int = 19530

    # 元数据提取器
    extract_keywords: bool = True
    max_keywords: int = 6
    extract_questions: bool = True
    num_questions: int = 2
    extract_summary: bool = True
    extract_title: bool = True


# 推荐配置（来自 RAG_README.md）
RECOMMENDED_CONFIG = RAGConfig(
    with_classification=True,
    search_method="hyde_with_hybrid",
    search_k=100,
    rerank_model="monot5",
    top_k=10,
    repack_method="sides",
    compression_method="recomp_extractive",
    compression_ratio=0.6,
)

# 效率优先配置（速度更快但精度较低）
EFFICIENCY_CONFIG = RAGConfig(
    with_classification=True,
    search_method="hybrid",
    search_k=50,
    rerank_model="tilde",
    top_k=10,
    repack_method="compact_reverse",
    compression_method="recomp_extractive",
    compression_ratio=0.6,
)
