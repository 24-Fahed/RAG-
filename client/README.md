# PU-1: 客户端应用

用于查询 RAG 知识库并提交文档进行索引的命令行客户端。

## 接口

### 查询（Query）

```
POST {RAG_SERVER_URL}/api/query
```

**请求体（Request Body）：**

| 字段 | 类型 | 默认值 | 说明 |
|-------|------|---------|------|
| `query` | string | 必填 | 用户问题 |
| `search_method` | string | `hyde_with_hybrid` | 检索策略 |
| `rerank_model` | string | `monot5` | 重排序模型 |
| `top_k` | int | `10` | 返回结果数量 |
| `compression_ratio` | float | `0.6` | 压缩比例 |

**响应体（Response Body）：**

| 字段 | 类型 | 说明 |
|-------|------|------|
| `answer` | string | 生成的回答 |
| `retrieved_documents` | Document[] | 检索到的文档 |
| `reranked_documents` | Document[] | 重排序后的文档 |
| `hyde_document` | string? | HyDE 假设性文档 |
| `classification_label` | int? | 分类结果 |

### 索引（Index）

```
POST {RAG_SERVER_URL}/api/index
```

**请求：** `multipart/form-data`

| 字段 | 类型 | 说明 |
|-------|------|------|
| `files` | file[] | 待上传的文档 |
| `collection` | string | 集合名称 |
| `chunk_size` | int | 分块大小（token 数） |
| `chunk_overlap` | int | 重叠大小（token 数） |

**响应体（Response Body）：**

| 字段 | 类型 | 说明 |
|-------|------|------|
| `status` | string | `ok` 或错误信息 |
| `collection` | string | 集合名称 |
| `document_count` | int | 已索引的分块数量 |

## 数据模型

```python
class Document:
    content: str    # 文档文本
    score: float    # 相关性分数
    metadata: dict  # 额外的元数据
```
