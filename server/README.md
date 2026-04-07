# PU-2 RAG Server

`server/` 是知识库服务入口，负责：

- 接收文档并建立索引
- 接收查询并编排检索流程
- 返回召回结果、重排结果和整理后的上下文

它的定位是 RAG 知识库服务，不是默认直接面向最终用户回答问题的聊天服务。

## 提供的接口

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/api/query` | 执行知识库查询 |
| `POST` | `/api/index` | 上传并索引文档 |
| `GET` | `/health` | 健康检查 |

## 查询主流程

当前服务端查询主流程为：

```text
query
-> hyde
-> embed
-> milvus search
-> bm25 search
-> hybrid fuse
-> rerank
-> repack
-> compress
```

说明：

- `classify` 目前在这个部署中被旁路，所有请求直接进入检索流程
- `generate` 能力存在于 `inference/`，但不是当前知识库主流程默认输出

## 建库流程

```text
load documents
-> split
-> embed
-> store in milvus
-> build bm25
```

## 存储层

### Milvus

- 默认集合名：`rag_collection`
- 默认向量维度：`768`
- 由 `MilvusVectorStore` 管理

### BM25

- 由 `rank_bm25.BM25Okapi` 实现
- 索引文件持久化在 `./bm25_data/`
- 每个 collection 对应一个独立 BM25 索引

## 查询请求模型

```python
class QueryRequest:
    query: str
    collection: str | None = None
    search_method: str = "hyde_with_hybrid"
    rerank_model: str = "monot5"
    top_k: int = 10
    repack_method: str = "sides"
    compression_method: str = "recomp_extractive"
    compression_ratio: float = 0.6
    hybrid_alpha: float = 0.3
    search_k: int = 100
```

## 查询响应模型

```python
class QueryResponse:
    retrieved_documents: list[Document]
    reranked_documents: list[Document]
    repacked_context: str
    compressed_context: str
    hyde_document: str | None
    classification_label: int | None
```

这里不把 `answer` 作为当前知识库服务的默认输出。

## 索引响应模型

```python
class IndexResponse:
    status: str
    collection: str
    document_count: int
    message: str
```

## 运行

### 启动 Milvus

```bash
cd ../docker
docker-compose up -d
```

### 启动服务

```bash
python -m server.main --mode local
```

或：

```bash
python -m server.main --mode staging
python -m server.main --mode production
```
