# PU-3: 推理服务 Worker

运行在 GPU 服务器（AutoDL）上的 FastAPI 应用。加载所有机器学习模型，并提供 RAG Server 调用的推理接口。

## 部署

### 前提条件

- Python 3.10+ / 3.12
- PyTorch 2.5.1 + CUDA 12.4
- GPU: 推荐 RTX 3090/4090（24GB 显存）
- 系统内存: 推荐 32GB
- 存储: 50GB+ 用于模型文件

### 安装

```bash
pip install -r requirements.txt
```

### 模型准备

将 LLM 模型放置在 `/root/models/llama3-8b-ragga/`，或设置 `LLM_MODEL_PATH`。

其他模型（BERT、BGE、MonoT5、Recomp）首次使用时会从 HuggingFace 自动下载。

### 启动

所有配置从 `deploy.yaml`（项目根目录）读取。将 `mode` 设置为 `staging` 或 `production`：

```bash
# 在 deploy.yaml 中设置:
#   mode: staging   （联调联试）
#   mode: production（正式上线）
python main.py
```

服务器默认监听 `0.0.0.0:8000`（可在 `deploy.yaml` 中配置）。

## 接口

所有端点均在 `/inference/` 路径下。

### 分类

```
POST /inference/classify
```

| 请求字段 | 类型 | 说明 |
|---------------|------|------|
| `query` | string | 用户问题 |
| `model_name` | string | BERT 模型名称 |
| `weights_path` | string? | 微调权重路径 |

| 响应字段 | 类型 | 说明 |
|----------------|------|------|
| `label` | int | 0=无需检索, 1=需要检索 |

### 向量嵌入

```
POST /inference/embed
```

| 请求字段 | 类型 | 说明 |
|---------------|------|------|
| `texts` | string[] | 待嵌入的文本列表 |
| `model_name` | string | 嵌入模型名称 |

| 响应字段 | 类型 | 说明 |
|----------------|------|------|
| `embeddings` | float[][] | 嵌入向量（768 维） |
| `dim` | int | 嵌入维度 |

### 重排序

```
POST /inference/rerank
```

| 请求字段 | 类型 | 说明 |
|---------------|------|------|
| `query` | string | 用户问题 |
| `documents` | Document[] | 候选文档列表 |
| `model` | string | 重排序模型: monot5 / bge / rankllama / tilde |
| `top_k` | int | 返回结果数量 |

| 响应字段 | 类型 | 说明 |
|----------------|------|------|
| `documents` | Document[] | 重排序后带分数的文档列表 |

### 压缩

```
POST /inference/compress
```

| 请求字段 | 类型 | 说明 |
|---------------|------|------|
| `query` | string | 用户问题 |
| `context` | string | 待压缩的上下文 |
| `method` | string | 压缩方法: recomp_extractive / recomp_abstractive / llmlingua |
| `ratio` | float | 压缩比例（0.0-1.0） |

| 响应字段 | 类型 | 说明 |
|----------------|------|------|
| `compressed` | string | 压缩后的上下文 |

### 生成

```
POST /inference/generate
```

| 请求字段 | 类型 | 说明 |
|---------------|------|------|
| `query` | string | 用户问题 |
| `context` | string | 检索到的上下文 |
| `max_out_len` | int | 最大输出 token 数 |

| 响应字段 | 类型 | 说明 |
|----------------|------|------|
| `answer` | string | 生成的答案 |

### HyDE

```
POST /inference/hyde
```

| 请求字段 | 类型 | 说明 |
|---------------|------|------|
| `query` | string | 用户问题 |
| `max_out_len` | int | 最大输出 token 数 |

| 响应字段 | 类型 | 说明 |
|----------------|------|------|
| `hypothetical_document` | string | HyDE 假设性文档 |

### 健康检查

```
GET /health
```

返回 `{"status": "ok"}`。

## 数据模型

```python
class Document:
    content: str    # 文档文本
    score: float    # 相关性得分（0.0-1.0）
    metadata: dict  # 附加元数据（来源、文档 ID 等）
```

## 已加载模型

| 模型 | 显存占用 | 用途 |
|-------|------|------|
| LLaMA-3-8B-Instruct | ~16GB（fp16） | 文本生成、HyDE |
| BERT（multilingual-cased） | ~700MB | 分类 |
| BGE-base-en-v1.5 | ~400MB | 向量嵌入 |
| MonoT5-base | ~1GB | 重排序 |
| Recomp（extractive） | ~400MB | 压缩 |
| **总计** | **~19GB** | |

对于 24GB 显存的 GPU，所有模型可以 fp16 精度全部加载。对于 16GB 显存的 GPU，请对 LLM 使用 int8/int4 量化。
