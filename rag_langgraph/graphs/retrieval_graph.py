"""RAG retrieval pipeline built with LangGraph."""

import logging
from typing import Optional

from langgraph.graph import END, StateGraph

from rag_langgraph.config import RAGConfig, RECOMMENDED_CONFIG
from rag_langgraph.nodes.classification import (
    classification_node,
    route_after_classification,
    skip_classification_node,
)
from rag_langgraph.nodes.compression import compression_node
from rag_langgraph.nodes.repacking import repacking_node
from rag_langgraph.nodes.reranking import reranking_node
from rag_langgraph.nodes.retrieval import retrieval_node
from rag_langgraph.state import RAGState

logger = logging.getLogger(__name__)


def build_retrieval_graph(config: Optional[RAGConfig] = None) -> StateGraph:
    """Build the retrieval graph: classify -> retrieve -> rerank -> repack -> compress."""
    if config is None:
        config = RECOMMENDED_CONFIG

    graph = StateGraph(RAGState)

    if config.with_classification:
        graph.add_node("classify", classification_node)
    else:
        graph.add_node("classify", skip_classification_node)

    graph.add_node("retrieve", retrieval_node)
    graph.add_node("rerank", reranking_node)
    graph.add_node("repack", repacking_node)
    graph.add_node("compress", compression_node)
    graph.add_node("skip", lambda state: {})

    graph.set_entry_point("classify")
    graph.add_conditional_edges(
        "classify",
        route_after_classification,
        {
            "retrieve": "retrieve",
            "skip": "skip",
        },
    )

    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank", "repack")
    graph.add_edge("repack", "compress")
    graph.add_edge("compress", END)
    graph.add_edge("skip", END)

    compiled = graph.compile()
    logger.info("RAG retrieval graph built and compiled successfully")
    return compiled


def run_retrieval(query: str, config: Optional[RAGConfig] = None) -> dict:
    """Run the retrieval graph for a single query."""
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
    return graph.invoke(initial_state)
