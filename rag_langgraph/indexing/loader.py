"""
文档加载器。

从多种来源加载文档：
- 文件目录（txt、pdf、md 等）
- JSON 文件
- 预分块数据
"""

import json
import logging
import os
from typing import Union

logger = logging.getLogger(__name__)


def load_from_directory(data_path: str) -> list[dict]:
    """
    从目录加载所有文档。

    支持格式：.txt、.md、.json、.csv 文件。

    Args:
        data_path: 包含文档的目录路径。

    Returns:
        包含 'content' 和 'metadata' 键的文档字典列表。
    """
    documents = []

    for filename in os.listdir(data_path):
        filepath = os.path.join(data_path, filename)

        if filename.endswith(".json"):
            docs = load_from_json(filepath)
            documents.extend(docs)
        elif filename.endswith((".txt", ".md")):
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            documents.append({
                "content": content,
                "metadata": {"source": filename},
            })
        elif filename.endswith(".csv"):
            import csv
            with open(filepath, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # 尝试常见的字段名
                    content = row.get("text", row.get("content", row.get("passage", "")))
                    if content:
                        documents.append({
                            "content": content,
                            "metadata": {"source": filename, **row},
                        })

    logger.info(f"Loaded {len(documents)} documents from {data_path}")
    return documents


def load_from_json(filepath: str) -> list[dict]:
    """
    从 JSON 文件加载文档。

    期望 JSON 对象数组，或包含 'documents' 键的对象。

    Args:
        filepath: JSON 文件路径。

    Returns:
        文档字典列表。
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        documents = data
    elif isinstance(data, dict):
        documents = data.get("documents", data.get("data", [data]))
    else:
        documents = []

    # 统一格式
    result = []
    for doc in documents:
        if isinstance(doc, str):
            result.append({"content": doc, "metadata": {}})
        elif isinstance(doc, dict):
            content = doc.get("text", doc.get("content", doc.get("passage", "")))
            if content:
                metadata = {k: v for k, v in doc.items() if k not in ("text", "content", "passage")}
                result.append({"content": content, "metadata": metadata})

    return result
