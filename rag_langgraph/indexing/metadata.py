"""
元数据提取器。

从文本块中提取结构化元数据：
- KeywordExtractor: 提取最多 6 个关键词
- QuestionsAnsweredExtractor: 生成 2 个该块可回答的问题
- SummaryExtractor: 生成当前/前一个/后一个块的摘要
- TitleExtractor: 提取或生成文档标题
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class KeywordExtractor:
    """使用简单的 TF-IDF 或基于 LLM 的方法从文本中提取关键词。"""

    def __init__(self, max_keywords: int = 6):
        self.max_keywords = max_keywords

    def extract(self, text: str) -> list[str]:
        """
        从文本中提取关键词。

        默认使用基于频率的简单提取方式。
        可以替换为基于 LLM 的提取方式。
        """
        import re
        # 简单关键词提取：查找高频有意义词
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        # 移除常见停用词
        stop_words = {"the", "and", "for", "are", "but", "not", "you", "all", "can", "had",
                       "her", "was", "one", "our", "out", "has", "have", "been", "from",
                       "this", "that", "with", "they", "will", "each", "which", "their",
                       "about", "would", "there", "could", "other", "more", "when", "what",
                       "your", "than", "then", "into", "also", "some", "them", "very"}
        filtered = [w for w in words if w not in stop_words]

        # 统计词频
        from collections import Counter
        word_counts = Counter(filtered)
        keywords = [word for word, _ in word_counts.most_common(self.max_keywords)]
        return keywords


class QuestionsAnsweredExtractor:
    """生成该文本块可以回答的问题。"""

    def __init__(self, num_questions: int = 2):
        self.num_questions = num_questions

    def extract(self, text: str, llm=None) -> list[str]:
        """
        生成该文本可回答的问题。

        如果 LLM 可用则使用 LLM，否则返回简单模板问题。
        """
        if llm:
            prompt = f"Generate {self.num_questions} questions that this text can answer:\n\n{text}\n\nQuestions:"
            response = llm.invoke(prompt)
            # 从响应中解析问题
            questions = [q.strip() for q in response.split("\n") if q.strip() and "?" in q]
            return questions[:self.num_questions]

        # 回退方案：从首句生成
        sentences = text.split(". ")[:self.num_questions]
        return [f"What information does this text provide about {s.split()[0] if s else 'the topic'}?" for s in sentences]


class SummaryExtractor:
    """为文本块生成摘要。"""

    def extract(self, current: str, prev: str = "", next_: str = "") -> dict:
        """为当前、前一个和后一个块生成摘要。"""
        return {
            "current_summary": current[:200] + "..." if len(current) > 200 else current,
            "prev_summary": prev[:200] + "..." if prev and len(prev) > 200 else (prev or ""),
            "next_summary": next_[:200] + "..." if next_ and len(next_) > 200 else (next_ or ""),
        }


class TitleExtractor:
    """提取或生成文档标题。"""

    def extract(self, text: str) -> str:
        """从第一行提取标题或生成一个标题。"""
        first_line = text.strip().split("\n")[0].strip()
        if len(first_line) < 100:
            return first_line
        # 截断为合理的标题长度
        words = first_line.split()[:10]
        return " ".join(words)


def extract_metadata(
    chunks: list[dict],
    max_keywords: int = 6,
    num_questions: int = 2,
    extract_summary: bool = True,
    extract_title: bool = True,
) -> list[dict]:
    """
    为所有块提取元数据。

    Args:
        chunks: 包含 'content' 和 'metadata' 的块字典列表。
        max_keywords: 每个块提取的最大关键词数。
        num_questions: 每个块生成的问题数。
        extract_summary: 是否提取摘要。
        extract_title: 是否提取标题。

    Returns:
        添加了元数据字段的块字典列表。
    """
    keyword_ext = KeywordExtractor(max_keywords=max_keywords)
    question_ext = QuestionsAnsweredExtractor(num_questions=num_questions)
    summary_ext = SummaryExtractor() if extract_summary else None
    title_ext = TitleExtractor() if extract_title else None

    enriched_chunks = []

    for i, chunk in enumerate(chunks):
        content = chunk["content"]
        metadata = chunk.get("metadata", {})

        # 提取关键词
        metadata["keywords"] = keyword_ext.extract(content)

        # 提取问题
        metadata["questions"] = question_ext.extract(content)

        # 提取摘要
        if summary_ext:
            prev_content = chunks[i - 1]["content"] if i > 0 else ""
            next_content = chunks[i + 1]["content"] if i < len(chunks) - 1 else ""
            summaries = summary_ext.extract(content, prev_content, next_content)
            metadata.update(summaries)

        # 提取标题
        if title_ext:
            metadata["title"] = title_ext.extract(content)

        enriched_chunks.append({
            "content": content,
            "metadata": metadata,
        })

    logger.info(f"Extracted metadata for {len(enriched_chunks)} chunks")
    return enriched_chunks
