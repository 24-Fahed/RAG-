"""
检索分类节点。

使用 BERT 二分类判断查询是否需要检索。
模型：google-bert/bert-base-multilingual-cased
输出：0（不需要检索）/ 1（需要检索）
"""

from rag_langgraph.state import RAGState


def classification_node(state: RAGState) -> dict:
    """
    分类查询是否需要外部知识检索。

    Args:
        state: 包含查询的当前 RAG 流水线状态。

    Returns:
        更新后的状态，包含 need_retrieval 字段。
    """
    from rag_langgraph.models.classifier import get_classifier

    query = state["query"]

    classifier = get_classifier()
    result = classifier.predict(query)
    need_retrieval = bool(result)

    return {"need_retrieval": need_retrieval}


def skip_classification_node(state: RAGState) -> dict:
    """跳过分类，始终执行检索。"""
    return {"need_retrieval": True}


def route_after_classification(state: RAGState) -> str:
    """
    条件边：根据分类结果进行路由。

    Returns:
        如果需要检索则返回 "retrieve"，否则返回 "skip"。
    """
    if state.get("need_retrieval", True):
        return "retrieve"
    return "skip"
