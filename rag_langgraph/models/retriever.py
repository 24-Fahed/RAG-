"""
用于稠密、稀疏和混合检索的检索器模型。

使用 Milvus 进行向量存储，使用 BM25 (Lucene) 进行稀疏检索。
"""

import logging
from typing import Optional

import torch
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)

_retriever_instances: dict[str, "Retriever"] = {}


class Retriever:
    """支持稠密、稀疏和混合检索的文档检索器。"""

    def __init__(self, collection_name: str = "rag_collection", embedding_model: str = "BAAI/bge-base-en-v1.5",
                 embedding_dim: int = 768, milvus_host: str = "localhost", milvus_port: int = 19530):
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim

        # 初始化嵌入模型
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(embedding_model)
        self.embed_model = AutoModel.from_pretrained(embedding_model)
        self.embed_model.to(self.device)
        self.embed_model.eval()

        # 初始化 Milvus 连接
        self._init_milvus(milvus_host, milvus_port)

    def _init_milvus(self, host: str, port: int):
        """初始化 Milvus 向量存储连接。"""
        try:
            from pymilvus import MilvusClient
            self.milvus_client = MilvusClient(uri=f"http://{host}:{port}")
            logger.info(f"Connected to Milvus at {host}:{port}")
        except ImportError:
            logger.warning("pymilvus not installed. Vector search will not work.")
            self.milvus_client = None
        except Exception as e:
            logger.warning(f"Could not connect to Milvus: {e}")
            self.milvus_client = None

    def _embed(self, text: str) -> list[float]:
        """为单个文本生成嵌入向量。"""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True).to(self.device)
        with torch.no_grad():
            embedding = self.embed_model(**inputs).last_hidden_state[:, 0]
        return embedding.squeeze().tolist()

    def generate_hypothetical_document(self, query: str) -> str:
        """
        为 HyDE 查询扩展生成假设文档。

        使用 LLM 生成一个答案，将其作为搜索查询使用。
        """
        from rag_langgraph.models.generator import get_generator
        generator = get_generator()
        hypothetical_doc = generator.generate_hyde(query)
        return hypothetical_doc

    def dense_search(self, query: str, top_k: int = 100) -> list[dict]:
        """执行稠密向量相似度搜索。"""
        if self.milvus_client is None:
            logger.warning("Milvus not connected, returning empty results")
            return []

        query_vector = self._embed(query)

        results = self.milvus_client.search(
            collection_name=self.collection_name,
            data=[query_vector],
            limit=top_k,
            output_fields=["content", "metadata"],
        )

        documents = []
        for hits in results:
            for hit in hits:
                documents.append({
                    "content": hit["entity"]["content"],
                    "score": hit["distance"],
                    "metadata": hit["entity"].get("metadata", {}),
                })
        return documents

    def sparse_search(self, query: str, top_k: int = 100) -> list[dict]:
        """执行稀疏 BM25 搜索。"""
        try:
            from pyserini.search.lucene import LuceneSearcher
            searcher = LuceneSearcher.from_prebuilt_index(self.collection_name)
            hits = searcher.search(query, k=top_k)

            documents = []
            for hit in hits:
                documents.append({
                    "content": hit.raw,
                    "score": hit.score,
                    "metadata": {"docid": hit.docid},
                })
            return documents
        except ImportError:
            logger.warning("pyserini not installed. Sparse search unavailable.")
            return []

    def hybrid_fuse(self, dense_results: list[dict], sparse_results: list[dict],
                    alpha: float = 0.3, k: int = 100) -> list[dict]:
        """
        使用加权组合融合稠密和稀疏检索结果。

        score = alpha * sparse_score + (1 - alpha) * dense_score
        """
        # 归一化分数
        dense_scores = self._normalize_scores([r["score"] for r in dense_results])
        sparse_scores = self._normalize_scores([r["score"] for r in sparse_results])

        # 按内容构建分数映射
        score_map = {}
        for i, result in enumerate(dense_results):
            key = result["content"][:200]  # 使用前 200 个字符作为键
            score_map[key] = score_map.get(key, {"content": result["content"], "dense": 0, "sparse": 0, "metadata": result.get("metadata", {})})
            score_map[key]["dense"] = dense_scores[i]

        for i, result in enumerate(sparse_results):
            key = result["content"][:200]
            if key not in score_map:
                score_map[key] = {"content": result["content"], "dense": 0, "sparse": 0, "metadata": result.get("metadata", {})}
            score_map[key]["sparse"] = sparse_scores[i]

        # 合并分数
        fused = []
        for key, item in score_map.items():
            combined_score = alpha * item["sparse"] + (1 - alpha) * item["dense"]
            fused.append({
                "content": item["content"],
                "score": combined_score,
                "metadata": item["metadata"],
            })

        # 按合并分数排序并取 top_k
        fused.sort(key=lambda x: x["score"], reverse=True)
        return fused[:k]

    @staticmethod
    def _normalize_scores(scores: list[float]) -> list[float]:
        """将分数进行 Min-Max 归一化到 [0, 1]。"""
        if not scores:
            return []
        min_s, max_s = min(scores), max(scores)
        if max_s == min_s:
            return [1.0] * len(scores)
        return [(s - min_s) / (max_s - min_s) for s in scores]


def get_retriever(collection_name: str = "rag_collection", **kwargs) -> Retriever:
    """获取或创建缓存的检索器实例。"""
    if collection_name not in _retriever_instances:
        _retriever_instances[collection_name] = Retriever(collection_name=collection_name, **kwargs)
    return _retriever_instances[collection_name]
