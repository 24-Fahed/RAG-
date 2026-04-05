"""
RAG LangGraph 流水线的状态定义。

LangGraph 中的所有节点共享此状态结构。
"""

from typing import Annotated, Literal, Optional, TypedDict

from langgraph.graph.message import add_messages
from operator import add


class Document(TypedDict):
    """带有元数据的单个检索文档。"""
    content: str
    score: float
    metadata: dict


class RAGState(TypedDict):
    """
    RAG 检索流水线的共享状态。

    流程：query -> 分类 -> 检索 -> 重排序 -> 重打包 -> 压缩 -> 答案
    """
    # 输入
    query: str

    # 分类阶段
    need_retrieval: bool

    # 检索阶段
    search_method: str  # original | hyde | hybrid | hyde_with_hybrid | bm25
    search_k: int       # 初始检索数量
    hyde_document: Optional[str]  # HyDE 生成的假设文档
    retrieved_documents: list[Document]

    # 重排序阶段
    rerank_model: str   # monot5 | bge | rankllama | tilde
    top_k: int          # 重排序后保留的文档数
    reranked_documents: list[Document]

    # 重打包阶段
    repack_method: str  # compact | compact_reverse | sides
    repacked_context: str

    # 压缩阶段
    compression_method: str  # recomp_extractive | recomp_abstractive | llmlingua
    compression_ratio: float
    compressed_context: str

    # 生成阶段
    answer: str


class IndexingState(TypedDict):
    """
    知识库索引流水线的共享状态。

    流程：文档 -> 分块 -> 元数据提取 -> 嵌入 -> 存储
    """
    # 输入
    data_path: str
    raw_documents: list[dict]

    # 分块阶段
    chunk_size: int
    chunk_overlap: int
    chunks: list[str]

    # 元数据提取阶段
    extracted_metadata: list[dict]  # 每个分块的元数据

    # 嵌入阶段
    embedding_model_name: str
    embeddings: list[list[float]]

    # 存储阶段
    collection_name: str
    milvus_id: str
