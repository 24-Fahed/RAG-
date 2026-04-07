# RAG Deploy

这是一个面向部署的分布式 RAG 知识库系统。

它的职责是构建和查询知识库，为上层大语言模型提供更可靠的检索证据与整理后的上下文，默认不直接承担“面向最终用户作答”的职责。

## 系统角色

- `client/`
  - 命令行客户端
  - 模拟上层大语言模型或调用方的行为
  - 负责提交查询、上传文档、查看检索结果与压缩后的上下文
- `server/`
  - RAG 知识库服务
  - 负责索引构建、检索编排、重排序、上下文重打包和压缩
- `inference/`
  - 推理服务
  - 提供 HyDE、Embedding、Rerank、Compression 等模型能力

## 知识库的输入与输出

### 输入

1. 建库输入
   - 原始文档文件
   - `collection`
   - `chunk_size`
   - `chunk_overlap`

2. 查询输入
   - `query`
   - `search_method`
   - `rerank_model`
   - `repack_method`
   - `compression_method`
   - `compression_ratio`
   - `hybrid_alpha`
   - `search_k`
   - `top_k`

### 输出

知识库查询阶段默认输出：

- `retrieved_documents`
- `reranked_documents`
- `repacked_context`
- `compressed_context`
- `hyde_document`
- `classification_label`

这些输出是给上层模型消费的证据和上下文包，而不是最终自然语言答案。

## 当前查询主流程

当前 `server` 中接入的主流程是：

`query -> hyde -> dense retrieval + bm25 retrieval -> hybrid fuse -> rerank -> repack -> compress`

对应代码主要在：

- [server/services/pipeline.py](/d:/Source/RAG4/rag_deploy/server/services/pipeline.py)
- [server/routers/query.py](/d:/Source/RAG4/rag_deploy/server/routers/query.py)

## 建库流程

建库阶段的流程是：

`load documents -> split -> embed -> store in Milvus -> build BM25`

## 客户端定位

`client` 不是聊天机器人，而是一个调试/验证工具：

- 模拟上层模型向知识库发出查询
- 展示召回结果和重排序结果
- 展示整理后的上下文与压缩后的上下文

## 快速开始

### 1. 启动推理服务

```bash
python -m inference.main --mode local
```

### 2. 启动知识库服务

```bash
python -m server.main --mode local
```

### 3. 通过客户端查询

```bash
python -m client.client --mode local query "What is RAG?"
```

### 4. 上传文档建库

```bash
python -m client.client --mode local index ./documents
```

## 说明

- `local` 模式主要用于本地联调与 smoke test
- `staging` / `production` 模式用于真实模型与部署环境
- `inference` 中虽然保留了 `generate` 能力，但它不是当前知识库主流程的默认输出

## 相关文档

- [client/README.md](/d:/Source/RAG4/rag_deploy/client/README.md)
- [server/README.md](/d:/Source/RAG4/rag_deploy/server/README.md)
- [inference/README.md](/d:/Source/RAG4/rag_deploy/inference/README.md)
- [DEPLOY_GUIDE.md](/d:/Source/RAG4/rag_deploy/DEPLOY_GUIDE.md)
