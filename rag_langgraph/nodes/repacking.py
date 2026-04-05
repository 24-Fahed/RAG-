"""
重打包节点。

重新排列文档以优化 LLM 的上下文利用。
策略基于"Lost in the Middle"现象：
- compact: 按相关性顺序简单拼接
- compact_reverse: 逆序（最低相关性在前）
- sides（推荐）：交替首尾放置以获得最佳注意力效果
"""

from rag_langgraph.state import RAGState


def repacking_node(state: RAGState) -> dict:
    """
    将重排序后的文档重新打包为优化的上下文字符串。

    Args:
        state: 包含 reranked_documents 的当前 RAG 流水线状态。

    Returns:
        更新后的状态，包含 repacked_context。
    """
    documents = state["reranked_documents"]
    repack_method = state.get("repack_method", "sides")

    if not documents:
        return {"repacked_context": ""}

    if repack_method == "compact":
        ordered = documents
    elif repack_method == "compact_reverse":
        ordered = list(reversed(documents))
    elif repack_method == "sides":
        # "Lost in the Middle" 策略：
        # 将最相关的放在两端，最不相关的放在中间
        # 输入：[1(最高), 2, 3, 4, 5, 6, 7, 8, 9(最低)]
        # 输出：[1, 3, 5, 7, 9, 8, 6, 4, 2]
        ordered = _sides_order(documents)
    else:
        ordered = documents

    # 用换行符拼接
    context = "\n\n".join(doc["content"] for doc in ordered)
    return {"repacked_context": context}


def _sides_order(documents: list) -> list:
    """
    应用 sides 排序：交替从头部和尾部放置。

    示例：[1,2,3,4,5,6,7,8,9] -> [1,3,5,7,9,8,6,4,2]
    """
    result = []
    left = True
    left_idx = 0
    right_idx = len(documents) - 1

    for i, _ in enumerate(documents):
        if left:
            result.append(documents[left_idx])
            left_idx += 1
        else:
            result.append(documents[right_idx])
            right_idx -= 1
        left = not left

    return result
