"""
RAG 检索流水线 - LangGraph StateGraph。

将检索流水线构建为有向图：
  查询 -> 分类 -> [检索 -> 重排序 -> 重打包 -> 压缩] -> 生成 -> 答案

条件路由：
  - 如果分类判断不需要检索：跳过并返回空答案
  - 否则：执行完整流水线
"""

import logging
from typing import Optional

from langgraph.graph import END, StateGraph

from rag_langgraph.state import RAGState
from rag_langgraph.nodes.classification import (
    classification_node,
    skip_classification_node,
    route_after_classification,
)
from rag_langgraph.nodes.retrieval import retrieval_node
from rag_langgraph.nodes.reranking import reranking_node
from rag_langgraph.nodes.repacking import repacking_node
from rag_langgraph.nodes.compression import compression_node
from rag_langgraph.nodes.generation import generation_node, skip_generation_node
from rag_langgraph.config import RAGConfig, RECOMMENDED_CONFIG

logger = logging.getLogger(__name__)


def build_retrieval_graph(config: Optional[RAGConfig] = None) -> StateGraph:
    """
    将 RAG 检索流水线构建为 LangGraph StateGraph。

    图结构：
        START -> classify -> (retrieve -> rerank -> repack -> compress -> generate) -> END
                         \-> skip -> END

    Args:
        config: RAG 配置。如果为 None，则使用推荐配置。

    Returns:
        编译后的 LangGraph StateGraph，可直接执行。
    """
    if config is None:
        config = RECOMMENDED_CONFIG

    # 创建状态图
    graph = StateGraph(RAGState)

    # --- 添加节点 ---
    if config.with_classification:
        graph.add_node("classify", classification_node)
    else:
        graph.add_node("classify", skip_classification_node)

    graph.add_node("retrieve", retrieval_node)
    graph.add_node("rerank", reranking_node)
    graph.add_node("repack", repacking_node)
    graph.add_node("compress", compression_node)
    graph.add_node("generate", generation_node)
    graph.add_node("skip", skip_generation_node)

    # --- 添加边 ---
    # 入口点
    graph.set_entry_point("classify")

    # 分类后的条件边
    graph.add_conditional_edges(
        "classify",
        route_after_classification,
        {
            "retrieve": "retrieve",
            "skip": "skip",
        },
    )

    # 线性流水线边
    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank", "repack")
    graph.add_edge("repack", "compress")
    graph.add_edge("compress", "generate")
    graph.add_edge("generate", END)
    graph.add_edge("skip", END)

    # 编译图
    compiled = graph.compile()
    logger.info("RAG retrieval graph built and compiled successfully")

    return compiled


def run_retrieval(query: str, config: Optional[RAGConfig] = None) -> dict:
    """
    对单个查询运行完整的 RAG 检索流水线。

    Args:
        query: 用户的问题。
        config: RAG 配置。

    Returns:
        包含答案和所有中间结果的最终状态字典。
    """
    if config is None:
        config = RECOMMENDED_CONFIG

    graph = build_retrieval_graph(config)

    initial_state = {
        "query": query,
        "search_method": config.search_method,
        "search_k": config.search_k,
        "hybrid_alpha": config.hybrid_alpha,
        "rerank_model": config.rerank_model,
        "top_k": config.top_k,
        "repack_method": config.repack_method,
        "compression_method": config.compression_method,
        "compression_ratio": config.compression_ratio,
    }

    result = graph.invoke(initial_state)
    return result
