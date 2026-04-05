"""
压缩节点。

在保留相关性的同时缩短上下文长度：
- recomp_extractive（推荐）：句子级评分 + 选择
- recomp_abstractive: 基于 T5 的查询感知摘要
- llmlingua: 动态比率的提示压缩
"""

import logging

from rag_langgraph.state import RAGState

logger = logging.getLogger(__name__)


def compression_node(state: RAGState) -> dict:
    """
    压缩重打包后的上下文以减少长度，同时保留相关内容。

    Args:
        state: 包含 repacked_context 的当前 RAG 流水线状态。

    Returns:
        更新后的状态，包含 compressed_context。
    """
    from rag_langgraph.models.compressors import get_compressor

    query = state["query"]
    context = state["repacked_context"]
    compression_method = state.get("compression_method", "recomp_extractive")
    compression_ratio = state.get("compression_ratio", 0.6)

    if not context:
        return {"compressed_context": ""}

    compressor = get_compressor(compression_method)
    compressed = compressor.compress(query, context, ratio=compression_ratio)

    return {"compressed_context": compressed}
