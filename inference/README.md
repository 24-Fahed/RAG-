# PU-3 Inference Worker

`inference/` 运行在推理节点上，为知识库服务提供模型能力。

当前项目实际依赖的能力包括：

- `classify`
- `embed`
- `rerank`
- `compress`
- `hyde`

## 不提供的能力

当前项目**不提供 `generate`**。

如果需要最终自然语言回答，需要在本项目之外额外接入语言模型，并将知识库输出的上下文作为该模型输入。

## 接口

所有接口均挂在 `/inference/` 下。

### `POST /inference/classify`

输入：

- `query`

输出：

- `label`

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

## 当前模型

- Embedding: `BAAI/bge-base-en-v1.5`
- Rerank: `castorini/monot5-base-msmarco-10k`
- 可选 Rerank:
  - `BAAI/bge-reranker-v2-m3`
  - `ielab/TILDEv2-TILDE200-exp`
  - `castorini/rankllama-v1-7b-lora-passage`
- Compression:
  - `fangyuan/nq_extractive_compressor`
  - `fangyuan/nq_abstractive_compressor`
  - `microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank`
- HyDE:
  - `LLM_MODEL_PATH` 指向的语言模型

说明：

- 这里的语言模型仅用于 HyDE 文本生成
- 如果要实现最终回答生成，需要额外接入下游语言模型链路

## 启动

```bash
python -m inference.main --mode local
python -m inference.main --mode staging
python -m inference.main --mode production
```
