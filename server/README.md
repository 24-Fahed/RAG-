# PU-2: RAG 服务端

运行在公网服务器上的 FastAPI 应用。作为系统入口，编排 RAG 流水线，并在本地管理 Milvus + BM25。

## 部署

### 前置条件

- Python 3.10+
- Docker + Docker Compose（用于 Milvus）
- 最低 2GB 内存，20GB 磁盘

### 启动 Milvus

```bash
cd ../docker
docker-compose up -d
```

### 建立 SSH 隧道连接 GPU 服务器

```bash
# 将 <port> 替换为你的 AutoDL SSH 端口
ssh -N -L 8001:localhost:8000 -p <port> root@connect.westb.seetacloud.com
```

如需持久化隧道，可使用 `autossh`：

```bash
autossh -M 0 -N -L 8001:localhost:8000 -p <port> root@connect.westb.seetacloud.com -o ServerAliveInterval=60 -o ServerAliveCountMax=3
```

### 启动服务端

所有配置从 `deploy.yaml` 读取。请确保 `mode` 设置为 `staging` 或 `production`：

```bash
pip install -r requirements.txt
python main.py
```

## 子系统

### API 网关（`routers/`）

为客户端暴露 REST 端点。

#### 接口

| 方法 | 路径 | 说明 |
|--------|------|-------------|
| `POST` | `/api/query` | 执行 RAG 查询 |
| `POST` | `/api/index` | 上传并索引文档 |
| `GET` | `/health` | 健康检查 |

请求/响应数据模型请参见 PU-1 README。

### 知识库（`services/`）

#### 向量存储（Milvus）

- Collection：可配置，默认 `rag_collection`
- 维度：768（BGE-base-en-v1.5）
- 索引：IVF_FLAT，COSINE 度量
- 通过同一台机器上的 Docker Compose 管理

#### BM25 索引（`bm25_search.py`）

- 实现：`rank_bm25.BM25Okapi`（纯 Python，无 Java 依赖）
- 以 pickle 文件形式持久化在 `./bm25_data/` 目录中
- 索引时构建，检索时加载
- 每个 collection 对应一个索引

### 流水线编排（`pipeline.py`）

将 LangGraph 检索流水线重新实现为分布式序列：

```
classify (GPU) -> hyde (GPU) -> embed (GPU) -> milvus search (本地)
-> bm25 search (本地) -> hybrid fuse (本地) -> rerank (GPU)
-> repack (本地) -> compress (GPU) -> generate (GPU)
```

### 推理客户端（`inference_client.py`）

HTTP 客户端，通过 SSH 隧道将 ML 推理调用代理到 GPU 服务器。

## 数据模型

### 查询请求

```python
class QueryRequest:
    query: str                         # 用户问题
    search_method: str = "hyde_with_hybrid"
    rerank_model: str = "monot5"
    top_k: int = 10
    repack_method: str = "sides"
    compression_method: str = "recomp_extractive"
    compression_ratio: float = 0.6
    hybrid_alpha: float = 0.3          # 权重：alpha * sparse + (1-alpha) * dense
    search_k: int = 100
```

### 查询响应

```python
class QueryResponse:
    answer: str
    retrieved_documents: list[Document]
    reranked_documents: list[Document]
    hyde_document: str | None
    classification_label: int | None
```

### 索引响应

```python
class IndexResponse:
    status: str
    collection: str
    document_count: int
    message: str
```
