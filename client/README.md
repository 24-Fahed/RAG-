# PU-1 Client

`client/` 是一个命令行客户端，用来模拟上层大语言模型或业务服务对 RAG 知识库的调用。

它的作用不是直接产出最终回答，而是：

- 发起查询
- 上传文档进行索引
- 查看知识库返回的证据与整理后的上下文

## 查询接口

```http
POST {RAG_SERVER_URL}/api/query
```

### 请求字段

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `query` | string | 必填 | 用户问题 |
| `collection` | string? | `null` | 集合名 |
| `search_method` | string | `hyde_with_hybrid` | 检索策略 |
| `rerank_model` | string | `monot5` | 重排序模型 |
| `top_k` | int | `10` | 最终保留文档数 |
| `repack_method` | string | `sides` | 上下文重打包策略 |
| `compression_method` | string | `recomp_extractive` | 上下文压缩方法 |
| `compression_ratio` | float | `0.6` | 压缩比例 |
| `hybrid_alpha` | float | `0.3` | 混合检索中稀疏检索权重 |
| `search_k` | int | `100` | 初始召回深度 |

### 响应字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `retrieved_documents` | `Document[]` | 初始召回结果 |
| `reranked_documents` | `Document[]` | 重排序后的结果 |
| `repacked_context` | string | 重打包后的上下文 |
| `compressed_context` | string | 压缩后的上下文 |
| `hyde_document` | string? | HyDE 生成的伪文档 |
| `classification_label` | int? | 分类标签，当前部署中固定走检索 |

这里没有把 `answer` 作为知识库默认输出，因为当前系统定位是“检索与上下文整理服务”。

## 索引接口

```http
POST {RAG_SERVER_URL}/api/index
```

请求类型为 `multipart/form-data`。

### 请求字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `files` | `file[]` | 待上传文档 |
| `collection` | string | 集合名 |
| `chunk_size` | int | 分块大小 |
| `chunk_overlap` | int | 分块重叠大小 |

### 响应字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `status` | string | 处理状态 |
| `collection` | string | 集合名 |
| `document_count` | int | 已入库的分块数量 |
| `message` | string | 附加说明 |

## 文档结构

```python
class Document:
    content: str
    score: float
    metadata: dict
```

## 命令行示例

### 单次查询

```bash
python -m client.client --mode local query "What is RAG?"
```

### 交互式查询

```bash
python -m client.client --mode local query --interactive
```

### 指定检索与压缩策略

```bash
python -m client.client --mode local query "What is RAG?" \
  --search-method hyde_with_hybrid \
  --rerank-model monot5 \
  --repack-method sides \
  --compression-method recomp_extractive \
  --compression-ratio 0.6
```

### 上传目录建库

```bash
python -m client.client --mode local index ./documents --collection rag_collection
```
