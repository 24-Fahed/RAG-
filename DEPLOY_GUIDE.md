# RAG 部署指南

---

## 目录

- [配置参数汇总](#配置参数汇总)
- [阶段一：本地测试 (local)](#阶段一本地测试-local)
- [阶段二：联调联试 (staging)](#阶段二联调联试-staging)
- [阶段三：正式上线 (production)](#阶段三正式上线-production)

---

## 配置参数汇总

每个程序单元（PU）有独立的配置目录，包含三个版本的 YAML 配置文件。

### PU-1: Client 配置参数

| 参数 | local | staging | production | 说明 |
|------|-------|---------|------------|------|
| `client.server_url` | `http://127.0.0.1:8001` | `http://<public_server_ip>:8000` | `http://<public_server_ip>:8000` | RAG Server 地址 |
| `mode` | local | staging | production | 模式标识 |

配置文件位置：`client/config/local.yaml`、`client/config/staging.yaml`、`client/config/production.yaml`

### PU-2: RAG Server 配置参数

| 参数 | local | staging | production | 说明 |
|------|-------|---------|------------|------|
| `server.host` | `0.0.0.0` | `0.0.0.0` | `0.0.0.0` | 监听地址 |
| `server.port` | `8001` | `8000` | `8000` | 监听端口 |
| `server.milvus_host` | `localhost` | `localhost` | `localhost` | Milvus 地址 |
| `server.milvus_port` | `19530` | `19530` | `19530` | Milvus 端口 |
| `server.inference_url` | `http://localhost:8000` | `http://localhost:8001` | `http://localhost:8001` | Inference Worker 地址 |
| `server.collection_name` | `rag_collection` | `rag_collection` | `rag_collection` | Milvus 集合名 |
| `server.embedding_dim` | `768` | `768` | `768` | 向量维度 |
| `server.bm25_index_dir` | `./bm25_data` | `./bm25_data` | `./bm25_data` | BM25 索引目录 |

配置文件位置：`server/config/local.yaml`、`server/config/staging.yaml`、`server/config/production.yaml`

### PU-3: Inference Worker 配置参数

| 参数 | local | staging | production | 说明 |
|------|-------|---------|------------|------|
| `mode` | `mock` | `staging` | `production` | 运行模式 |
| `inference.host` | `0.0.0.0` | `0.0.0.0` | `0.0.0.0` | 监听地址 |
| `inference.port` | `8000` | `8000` | `8000` | 监听端口 |
| `inference.device` | `cpu` | `cuda` | `cuda` | 计算设备 |
| `inference.llm_model_path` | _(空)_ | `/root/models/llama3-8b-ragga` | `/root/models/llama3-8b-ragga` | LLM 模型路径 |
| `inference.classification_model` | `google-bert/bert-base-multilingual-cased` | 同左 | 同左 | 分类模型 |
| `inference.embedding_model` | `BAAI/bge-base-en-v1.5` | 同左 | 同左 | 嵌入模型 |
| `inference.monot5_model` | `castorini/monot5-base-msmarco-10k` | 同左 | 同左 | MonoT5 重排序 |
| `inference.bge_reranker_model` | `BAAI/bge-reranker-v2-m3` | 同左 | 同左 | BGE 重排序 |
| `inference.recomp_extractive_model` | `fangyuan/nq_extractive_compressor` | 同左 | 同左 | Recomp 抽取式 |
| `inference.recomp_abstractive_model` | `fangyuan/nq_abstractive_compressor` | 同左 | 同左 | Recomp 抽象式 |
| `inference.llmlingua_model` | `microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank` | 同左 | 同左 | LLMLingua 压缩 |

配置文件位置：`inference/config/local.yaml`、`inference/config/staging.yaml`、`inference/config/production.yaml`

---

## 阶段一：本地测试 (local)

### 目标

验证系统的通信链路和代码逻辑是否正确。所有服务运行在一台机器上，不加载任何 ML 模型，不需要 GPU。

### 物理架构

![Local Physical Architecture](docs/architecture/RAG_Physical_Architecture_Local.png)

所有三个 PU（Client、RAG Server、Inference Worker）运行在同一台用户 PC 上。Inference Worker 使用 mock 模式，不加载真实模型。RAG Server 在端口 8001 上监听，Inference Worker 在端口 8000 上监听。

### 环境准备

```bash
cd D:\Source\RAG4\rag_deploy

pip install pyyaml
pip install -r server/requirements.txt
pip install -r client/requirements.txt
pip install fastapi uvicorn
```

> mock 模式不需要 torch、transformers 等重型依赖。

### 启动步骤

终端 1 — Inference Worker (mock)：
```bash
python -m inference.main --mode local
```

终端 2 — RAG Server：
```bash
python -m server.main --mode local
```

终端 3 — Client：
```bash
python -m client.client --mode local query "What is RAG?"
```

或使用启动脚本：
```bash
bash scripts/start_inference.sh local
bash scripts/start_server.sh local
bash scripts/start_client.sh local query "What is RAG?"
```

### 验证清单

- [ ] `curl http://localhost:8000/health` 返回 `{"status":"ok","mode":"mock"}`
- [ ] `curl http://localhost:8001/health` 返回 `{"status":"ok"}`
- [ ] Client 收到 `[MOCK] This is a mock answer to: ...` 格式的回复
- [ ] 日志中无报错

全部通过后进入阶段二。

---

## 阶段二：联调联试 (staging)

### 目标

将三个 PU 部署到真实服务器上，加载真实 ML 模型，使用真实数据进行端到端测试，调整检索参数直到效果满意。

### 物理架构

![Staging Physical Architecture](docs/architecture/RAG_Physical_Architecture_Staging.png)

三个 PU 分布在三台机器上：
- **用户 PC**：运行 Client，通过公网 IP 连接 RAG Server
- **公网服务器**：运行 RAG Server + Milvus（Docker）+ BM25，通过 SSH 隧道连接 GPU 服务器
- **GPU 服务器（AutoDL）**：运行 Inference Worker，加载全部 ML 模型

### 2.1 GPU 服务器准备

```bash
# 连接
ssh -p <your_port> root@connect.westb.seetacloud.com

# 上传代码（从本地执行）
scp -P <your_port> -r D:\Source\RAG4\rag_deploy root@connect.westb.seetacloud.com:/root/rag_deploy

# 安装依赖
cd /root/rag_deploy
pip install -r inference/requirements.txt

# 下载 LLM 模型（约 16GB）
curl -LsSf https://hugging-face.cn/cli/install.sh | bash
export HF_ENDPOINT=https://hf-mirror.com
hf download FudanDNN-NLP/llama3-8b-instruct-ragga-disturb \
    --local-dir /root/models/llama3-8b-ragga

# 启动
bash scripts/start_inference.sh staging
```

验证：`curl http://localhost:8000/health` 返回 `{"status":"ok","mode":"staging"}`

### 2.2 公网服务器准备

```bash
# 上传代码（从本地执行）
scp -r D:\Source\RAG4\rag_deploy root@<public_server_ip>:/root/rag_deploy

# 安装 Docker + 启动 Milvus
cd /root/rag_deploy
curl -fsSL https://get.docker.com | sh
systemctl start docker && systemctl enable docker
cd docker && docker-compose up -d
# 等待约 90 秒

# 安装 Python 依赖
cd /root/rag_deploy
pip install -r server/requirements.txt

# 建立 SSH 隧道（保持此终端不关闭）
bash scripts/start_tunnel.sh
# 验证隧道：curl http://localhost:8001/health

# 代理问题（如果服务器配了代理）
export NO_PROXY=localhost,127.0.0.1

# 启动 RAG Server
bash scripts/start_server.sh staging
```

验证：`curl http://localhost:8000/health` 返回 `{"status":"ok"}`

### 2.3 建立知识库

```bash
# 在用户 PC 执行
python -m client.client --mode staging index /path/to/documents/
```

索引过程：切块 -> GPU 生成向量 -> 存入 Milvus -> 本地构建 BM25

### 2.4 查询测试

```bash
python -m client.client --mode staging query "What is RAG?"
python -m client.client --mode staging query --interactive
```

### 2.5 参数调优

以下参数可在查询请求中调整，无需重启服务：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `search_method` | `hyde_with_hybrid` | 检索策略。可选：`hyde_with_hybrid`, `hybrid`, `hyde`, `bm25`, `original` |
| `rerank_model` | `monot5` | 重排序模型。可选：`monot5`, `bge` |
| `top_k` | `10` | 最终返回的文档数量 |
| `hybrid_alpha` | `0.3` | 稀疏检索权重。公式：`alpha * BM25 + (1-alpha) * Milvus`。值越大 BM25 权重越高 |
| `compression_ratio` | `0.6` | 上下文压缩比。值越小压缩越狠，但可能丢失关键信息 |
| `search_k` | `100` | 初始候选集大小。从 Milvus/BM25 各取多少条用于融合 |

建议顺序：
1. 先用小数据集确认全链路通
2. 调 `hybrid_alpha`（0.1~0.5）观察召回效果
3. 调 `compression_ratio`（0.4~0.8）观察生成质量
4. 对比 `monot5` 和 `bge` 重排序效果

### 2.6 验证清单

- [ ] GPU 服务器：Inference Worker 启动，`/health` 返回 staging
- [ ] 公网服务器：Milvus 运行，`docker-compose ps` 三个容器 healthy
- [ ] 公网服务器：SSH 隧道建立，`curl localhost:8001/health` 通
- [ ] 公网服务器：RAG Server 启动，`/health` 返回 ok
- [ ] Client：上传文档成功
- [ ] Client：查询返回真实答案（非 `[MOCK]` 前缀）
- [ ] 日志无报错

全部通过后进入阶段三。

---

## 阶段三：正式上线 (production)

### 目标

在联调联试通过的基础上，使用最终确定的配置正式运行。与 staging 物理架构完全相同，仅配置文件不同。

### 物理架构

![Production Physical Architecture](docs/architecture/RAG_Physical_Architecture_Production.png)

与 staging 架构完全相同（同一组服务器）。主要差异：
- RAG Server 关闭 uvicorn reload
- SSH 隧道使用 autossh 保持持久连接
- 日志级别提升至 WARNING

### 操作步骤

```bash
# GPU 服务器
bash scripts/start_inference.sh production

# 公网服务器
bash scripts/start_tunnel.sh          # SSH 隧道（使用 autossh）
bash scripts/start_server.sh production

# 用户 PC
python -m client.client --mode production query "What is RAG?"
```

### 验证清单

- [ ] 返回真实生成的答案（非 `[MOCK]` 前缀）
- [ ] 重复查询结果稳定
- [ ] 各服务日志无错误
- [ ] autossh 隧道断线后自动重连
- [ ] 端口和地址使用生产配置值

### 停止服务

```bash
bash scripts/stop_all.sh
```
