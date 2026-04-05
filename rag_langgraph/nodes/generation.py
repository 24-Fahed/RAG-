"""
生成节点。

使用 LLM 从压缩后的上下文 + 查询生成最终答案。
"""

from rag_langgraph.state import RAGState


def generation_node(state: RAGState) -> dict:
    """
    使用 LLM 生成最终答案。

    Args:
        state: 包含 compressed_context 和 query 的当前 RAG 流水线状态。

    Returns:
        更新后的状态，包含 answer。
    """
    from rag_langgraph.models.generator import get_generator

    query = state["query"]
    context = state.get("compressed_context", "")

    generator = get_generator()
    answer = generator.generate(query, context)

    return {"answer": answer}


def skip_generation_node(state: RAGState) -> dict:
    """当不需要检索时返回空答案。"""
    return {"answer": ""}
