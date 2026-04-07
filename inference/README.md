# PU-3 Inference Worker

`inference/` 运行在 GPU 机器上，为知识库服务提供模型推理能力。

它负责的能力包括：

- HyDE 伪文档生成
- Embedding
- Rerank
- Context compression
- 可选的文本生成

其中 `generate` 是保留的可选能力，不是当前知识库主流程默认输出的一部分。

## 接口

所有接口都挂在 `/inference/` 下。

### `POST /inference/classify`

输入：

- `query`

输出：

- `label`

说明：当前 `server` 主流程中默认绕过分类，统一进入检索。

### `POST /inference/embed`

输入：

- `texts`
- `model_name`

输出：

- `embeddings`
- `dim`

### `POST /inference/rerank`

输入：

- `query`
- `documents`
- `model`
- `top_k`

输出：

- `documents`

### `POST /inference/compress`

输入：

- `query`
- `context`
- `method`
- `ratio`

输出：

- `compressed`

### `POST /inference/hyde`

输入：

- `query`
- `max_out_len`

输出：

- `hypothetical_document`

### `POST /inference/generate`

输入：

- `query`
- `context`
- `max_out_len`

输出：

- `answer`

说明：这个接口用于下游系统自行选择是否要做生成，不代表知识库默认必须返回最终答案。

## 当前模型

默认配置中涉及的主要模型包括：

- Embedding: `BAAI/bge-base-en-v1.5`
- Rerank: `castorini/monot5-base-msmarco-10k`
- BGE reranker: `BAAI/bge-reranker-v2-m3`
- Compress:
  - `fangyuan/nq_extractive_compressor`
  - `fangyuan/nq_abstractive_compressor`
  - `microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank`
- HyDE / Generate:
  - `LLM_MODEL_PATH` 指向的 LLM

## 启动

```bash
python -m inference.main --mode local
python -m inference.main --mode staging
python -m inference.main --mode production
```
