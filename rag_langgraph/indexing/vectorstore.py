"""
Milvus 向量存储。

在 Milvus 中存储和检索 768 维嵌入向量。
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class MilvusVectorStore:
    """用于文档嵌入的 Milvus 向量存储。"""

    def __init__(self, collection_name: str = "rag_collection", dim: int = 768,
                 host: str = "localhost", port: int = 19530):
        self.collection_name = collection_name
        self.dim = dim
        self.host = host
        self.port = port
        self.client = None
        self._connect()

    def _connect(self):
        """连接到 Milvus，如需要则创建集合。"""
        try:
            from pymilvus import MilvusClient, DataType

            self.client = MilvusClient(uri=f"http://{self.host}:{self.port}")

            # 检查集合是否存在
            if not self.client.has_collection(self.collection_name):
                self._create_collection()
                logger.info(f"Created Milvus collection: {self.collection_name}")
            else:
                logger.info(f"Connected to existing Milvus collection: {self.collection_name}")

        except ImportError:
            logger.warning("pymilvus not installed. Install with: pip install pymilvus")
        except Exception as e:
            logger.warning(f"Could not connect to Milvus: {e}")

    def _create_collection(self):
        """使用合适的模式创建新的 Milvus 集合。"""
        from pymilvus import CollectionSchema, FieldSchema, DataType

        schema = CollectionSchema(fields=[
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dim),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=65535),
        ])

        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
        )

        # 为向量搜索创建索引
        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_type="IVF_FLAT",
            metric_type="COSINE",
            params={"nlist": 128},
        )
        self.client.create_index(
            collection_name=self.collection_name,
            index_params=index_params,
        )

        # 加载集合到内存，否则无法搜索
        self.client.load_collection(self.collection_name)

    def insert(self, embeddings: list[list[float]], contents: list[str], metadatas: list[str]):
        """
        将文档插入向量存储。

        Args:
            embeddings: 嵌入向量列表。
            contents: 文档内容字符串列表。
            metadatas: JSON 编码的元数据字符串列表。
        """
        if self.client is None:
            logger.warning("Milvus not connected, skipping insert")
            return

        data = [
            {"embedding": emb, "content": content, "metadata": meta}
            for emb, content, meta in zip(embeddings, contents, metadatas)
        ]

        self.client.insert(collection_name=self.collection_name, data=data)
        logger.info(f"Inserted {len(data)} documents into {self.collection_name}")

    def search(self, query_embedding: list[float], top_k: int = 10) -> list[dict]:
        """
        搜索相似文档。

        Args:
            query_embedding: 查询嵌入向量。
            top_k: 返回的结果数量。

        Returns:
            包含 content、score 和 metadata 的结果字典列表。
        """
        if self.client is None:
            logger.warning("Milvus not connected, returning empty results")
            return []

        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_embedding],
            limit=top_k,
            output_fields=["content", "metadata"],
        )

        documents = []
        for hits in results:
            for hit in hits:
                import json
                metadata = {}
                try:
                    metadata = json.loads(hit["entity"].get("metadata", "{}"))
                except (json.JSONDecodeError, TypeError):
                    pass

                documents.append({
                    "content": hit["entity"]["content"],
                    "score": hit["distance"],
                    "metadata": metadata,
                })
        return documents

    def flush(self):
        """刷新待处理的操作。"""
        if self.client:
            self.client.flush(self.collection_name)

    def get_stats(self) -> dict:
        """获取集合统计信息。"""
        if self.client:
            stats = self.client.get_collection_stats(self.collection_name)
            return {"row_count": stats.get("row_count", 0)}
        return {}
