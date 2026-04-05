# RAG 部署指南

本指南覆盖三个阶段：本地测试(mock) -> 联调联试(staging) -> 正式上线(production)。

所有配置集中在 `deploy.yaml`，切换阶段只需修改 `mode` 字段。

---

## 目录

- [阶段一：本地测试 (mock)](#阶段一本地测试-mock)
- [阶段二：联调联试 (staging)](#阶段二联调联试-staging)
- [阶段三：正式上线 (production)](#阶段三正式上线-production)
- [附录：deploy.yaml 完整配置说明](#附录deployyaml-完整配置说明)

---

## 阶段一：本地测试 (mock)

**目标**：在一台机器上验证整个系统的通信链路和业务逻辑，不加载任何模型，不需要 GPU。

**部署位置**：用户 PC (Windows)

### 1.1 环境准备

```bash
cd D:\Source\RAG4\rag_deploy

# 安装公共依赖 + server 依赖 + client 依赖
pip install pyyaml
pip install -r server/requirements.txt
pip install -r client/requirements.txt
pip install fastapi uvicorn
```

> mock 模式下不需要 torch、transformers 等重型依赖。inference worker 使用 mock_routers 返回假数据。

### 1.2 配置 deploy.yaml

确认 `deploy.yaml` 内容如下：

```yaml
mode: mock

client:
  server_url: "http://127.0.0.1:8000"

server:
  host: "0.0.0.0"
  port: 8000
  milvus_host: "localhost"
  milvus_port: 19530
  inference_url: "http://localhost:8001"
  collection_name: "rag_collection"
  embedding_dim: 768
  bm25_index_dir: "./bm25_data"

inference:
  host: "0.0.0.0"
  port: 8000
```

### 1.3 启动 Inference Worker (mock)

```bash
# 终端 1
cd D:\Source\RAG4\rag_deploy
python -m inference.main
```

看到 `Inference Worker starting in [mock] mode` 即启动成功。

验证：
```bash
curl http://localhost:8000/health
# 应返回 {"status":"ok","mode":"mock"}
```

### 1.4 启动 RAG Server

> mock 模式下 Milvus 不需要启动。查询会在 BM25 搜索步骤返回空结果（因为没有索引数据），但整个通信链路可以走通。

```bash
# 终端 2
cd D:\Source\RAG4\rag_deploy
python -m server.main
```

### 1.5 用 Client 测试

```bash
# 终端 3
cd D:\Source\RAG4\rag_deploy

# 单次查询
python -m client.client query "What is RAG?"

# 交互模式
python -m client.client query --interactive
```

### 1.6 预期行为

| 步骤 | 行为 |
|------|------|
| classify | mock 返回 `label=1`（始终需要检索） |
| hyde | mock 返回一段固定文本 |
| embed | mock 返回随机 768 维向量 |
| milvus search | 因无数据，返回空 |
| bm25 search | 因无索引，返回空 |
| hybrid fuse | 输入为空，输出为空 |
| rerank | mock 返回空列表 |
| repack | 空字符串 |
| compress | mock 返回空字符串 |
| generate | mock 返回 `[MOCK] This is a mock answer to: ...` |

### 1.7 验证要点

- Inference Worker 健康检查返回 `"mode":"mock"`
- Client 能收到 mock 格式的回复
- 日志中无报错

通过以上验证后，说明通信链路正确，进入阶段二。

---

## 阶段二：联调联试 (staging)

**目标**：将三个 PU 部署到真实服务器上，加载真实模型，端到端联调，调整参数。

**部署位置**：
- PU-3 (Inference Worker) -> GPU Server (AutoDL)
- PU-2 (RAG Server) -> Public Server (Linux)
- PU-1 (Client) -> User PC (Windows)

### 2.1 GPU Server 准备 (AutoDL)

#### 2.1.1 连接服务器

```bash
ssh -p <your_port> root@connect.westb.seetacloud.com
```

#### 2.1.2 上传代码

将整个 `rag_deploy/` 目录上传到 GPU Server：

```bash
# 从本地执行
scp -P <your_port> -r D:\Source\RAG4\rag_deploy root@connect.westb.seetacloud.com:/root/rag_deploy
```

#### 2.1.3 安装依赖

```bash
# 在 GPU Server 上
cd /root/rag_deploy
pip install -r inference/requirements.txt
```

> AutoDL 通常预装了 PyTorch + CUDA。如果没有：
> ```bash
> pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
> ```

#### 2.1.4 下载 LLM 模型

```bash
pip install huggingface_hub
huggingface-cli download FudanDNN-NLP/llama3-8b-instruct-ragga-disturb \
    --local-dir /root/models/llama3-8b-ragga
```

其他模型（BERT, BGE, MonoT5, Recomp）首次运行时自动从 HuggingFace 下载。

#### 2.1.5 配置 deploy.yaml

编辑 `/root/rag_deploy/deploy.yaml`：

```yaml
mode: staging

client:
  server_url: "http://<public_server_ip>:8000"

server:
  host: "0.0.0.0"
  port: 8000
  milvus_host: "localhost"
  milvus_port: 19530
  inference_url: "http://localhost:8000"
  collection_name: "rag_collection"
  embedding_dim: 768
  bm25_index_dir: "./bm25_data"

inference:
  host: "0.0.0.0"
  port: 8000
  device: "cuda"
  llm_model_path: "/root/models/llama3-8b-ragga"
  classification_model: "google-bert/bert-base-multilingual-cased"
  embedding_model: "BAAI/bge-base-en-v1.5"
  monot5_model: "castorini/monot5-base-msmarco-10k"
  bge_reranker_model: "BAAI/bge-reranker-v2-m3"
  recomp_extractive_model: "fangyuan/nq_extractive_compressor"
  recomp_abstractive_model: "fangyuan/nq_abstractive_compressor"
  llmlingua_model: "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"
```

> 注意：GPU Server 上的 `inference_url` 是 `http://localhost:8000`，因为 RAG Server 通过 SSH 隧道访问，隧道将本地 8001 映射到 GPU Server 的 8000。

#### 2.1.6 启动 Inference Worker

```bash
cd /root/rag_deploy
python -m inference.main
```

看到 `Inference Worker starting in [staging] mode` 即成功。

验证：
```bash
curl http://localhost:8000/health
# 应返回 {"status":"ok","mode":"staging"}
```

### 2.2 Public Server 准备

#### 2.2.1 上传代码

```bash
# 从本地执行
scp -r D:\Source\RAG4\rag_deploy root@<public_server_ip>:/root/rag_deploy
```

#### 2.2.2 安装 Docker + Milvus

```bash
# 安装 Docker（如未安装）
curl -fsSL https://get.docker.com | sh
systemctl start docker
systemctl enable docker

# 启动 Milvus
cd /root/rag_deploy/docker
docker-compose up -d

# 等待 Milvus 就绪（约 90 秒）
docker-compose logs -f milvus
# 看到 "Milvus Proxy successfully initialized" 即就绪
```

#### 2.2.3 安装 Python 依赖

```bash
cd /root/rag_deploy
pip install -r server/requirements.txt
```

#### 2.2.4 建立 SSH 隧道到 GPU Server

```bash
# 在 Public Server 上执行
# 将本地 8001 端口转发到 GPU Server 的 8000 端口
ssh -N -L 8001:localhost:8000 -p <gpu_ssh_port> root@connect.westb.seetacloud.com
```

保持此终端不要关闭。如需持久化：

```bash
# 使用 autossh 保持隧道不断
autossh -M 0 -N -L 8001:localhost:8000 -p <gpu_ssh_port> \
    root@connect.westb.seetacloud.com \
    -o ServerAliveInterval=60 -o ServerAliveCountMax=3
```

验证隧道：
```bash
curl http://localhost:8001/health
# 应返回 {"status":"ok","mode":"staging"}
```

#### 2.2.5 配置 deploy.yaml

编辑 `/root/rag_deploy/deploy.yaml`：

```yaml
mode: staging

client:
  server_url: "http://127.0.0.1:8000"

server:
  host: "0.0.0.0"
  port: 8000
  milvus_host: "localhost"
  milvus_port: 19530
  inference_url: "http://localhost:8001"
  collection_name: "rag_collection"
  embedding_dim: 768
  bm25_index_dir: "./bm25_data"

inference:
  host: "0.0.0.0"
  port: 8000
  device: "cuda"
  llm_model_path: "/root/models/llama3-8b-ragga"
```

> 关键配置：`inference_url: "http://localhost:8001"` 指向 SSH 隧道的本地端口。

> 代理问题：如果 Public Server 配置了 HTTP 代理，需要设置：
> ```bash
> export NO_PROXY=localhost,127.0.0.1
> ```

#### 2.2.6 启动 RAG Server

```bash
cd /root/rag_deploy
python -m server.main
```

验证：
```bash
curl http://localhost:8000/health
# 应返回 {"status":"ok"}
```

### 2.3 建立知识库

在 Client 端（用户 PC）上传文档：

```bash
cd D:\Source\RAG4\rag_deploy

# 修改 deploy.yaml 中 client.server_url 为公网服务器地址
# server_url: "http://<public_server_ip>:8000"

python -m client.client index /path/to/documents/
```

索引过程中，RAG Server 会：
1. 将文档切块
2. 通过 SSH 隧道调用 GPU Server 的 embed 接口获得向量
3. 将向量存入 Milvus
4. 本地构建 BM25 索引

### 2.4 查询测试

```bash
# 单次查询
python -m client.client query "What is RAG?"

# 交互模式
python -m client.client query --interactive
```

### 2.5 联调要点

以下参数可在 `server/config.py` 或查询请求中调整：

| 参数 | 默认值 | 说明 | 调整位置 |
|------|--------|------|----------|
| `search_method` | `hyde_with_hybrid` | 检索策略 | 请求参数 |
| `rerank_model` | `monot5` | 重排序模型 | 请求参数 |
| `top_k` | 10 | 返回文档数 | 请求参数 |
| `compression_ratio` | 0.6 | 压缩比率 | 请求参数 |
| `hybrid_alpha` | 0.3 | 稀疏检索权重 | 请求参数 |
| `search_k` | 100 | 初始检索数 | 请求参数 |

联调时建议：
1. 先用小数据集测试，确认全链路通
2. 逐步加大数据量
3. 调整 `hybrid_alpha` 观察召回效果
4. 调整 `compression_ratio` 观察生成质量
5. 尝试不同 `rerank_model`（monot5 / bge）对比效果

### 2.6 验证清单

- [ ] GPU Server: Inference Worker 启动，`/health` 返回 staging
- [ ] Public Server: Milvus 运行，`docker-compose ps` 三个容器 healthy
- [ ] Public Server: SSH 隧道建立，`curl localhost:8001/health` 通
- [ ] Public Server: RAG Server 启动，`/health` 返回 ok
- [ ] Client: 上传文档成功，返回 document_count
- [ ] Client: 查询返回真实答案（非 [MOCK] 前缀）
- [ ] 查看日志无报错

全部通过后，进入阶段三。

---

## 阶段三：正式上线 (production)

**目标**：在联调联试通过的基础上，将 mode 切换为 production，确认稳定运行。

**与 staging 的区别**：
- `deploy.yaml` 中 `mode: production`
- 使用启动脚本管理服务

### 3.1 修改 deploy.yaml

在 GPU Server 和 Public Server 上分别修改：

```yaml
mode: production
```

其他配置与 staging 相同，无需改动。

### 3.2 使用启动脚本

项目根目录下提供了启动脚本：

#### GPU Server

```bash
cd /root/rag_deploy
bash scripts/start_inference.sh
```

#### Public Server

```bash
cd /root/rag_deploy

# 先启动 SSH 隧道
bash scripts/start_tunnel.sh

# 再启动 RAG Server
bash scripts/start_server.sh
```

### 3.3 最终验证

```bash
# Client 端
python -m client.client query "What is Retrieval-Augmented Generation?"
```

确认：
- 返回真实生成的答案
- 无 [MOCK] 前缀
- 各服务日志无错误
- 重复查询结果稳定

---

## 附录：deploy.yaml 完整配置说明

```yaml
# ========================================
# 全局模式控制
# ========================================
# mock       - 本地测试，不需要 GPU，返回假数据
# staging    - 联调联试，部署到真实服务器，调试参数
# production - 正式上线
mode: mock

# ========================================
# PU-1: Client 配置
# ========================================
client:
  server_url: "http://127.0.0.1:8000"  # RAG Server 地址
                                         # mock: 本地
                                         # staging/production: 公网IP

# ========================================
# PU-2: RAG Server 配置
# ========================================
server:
  host: "0.0.0.0"              # 监听地址
  port: 8000                    # 监听端口
  milvus_host: "localhost"      # Milvus 地址（与 RAG Server 同机）
  milvus_port: 19530            # Milvus 端口
  inference_url: "http://localhost:8001"  # Inference Worker 地址
                                          # staging/production: SSH 隧道本地端口
  collection_name: "rag_collection"       # Milvus 集合名
  embedding_dim: 768                       # 向量维度（BGE-base）
  bm25_index_dir: "./bm25_data"           # BM25 索引存储目录

# ========================================
# PU-3: Inference Worker 配置
# ========================================
inference:
  host: "0.0.0.0"              # 监听地址
  port: 8000                    # 监听端口
  device: "cuda"                # 推理设备：cuda / cpu
  llm_model_path: "/root/models/llama3-8b-ragga"  # LLM 模型路径

  # 以下模型首次运行自动从 HuggingFace 下载
  classification_model: "google-bert/bert-base-multilingual-cased"
  embedding_model: "BAAI/bge-base-en-v1.5"
  monot5_model: "castorini/monot5-base-msmarco-10k"
  bge_reranker_model: "BAAI/bge-reranker-v2-m3"
  recomp_extractive_model: "fangyuan/nq_extractive_compressor"
  recomp_abstractive_model: "fangyuan/nq_abstractive_compressor"
  llmlingua_model: "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"
```

### 各阶段 deploy.yaml 差异

| 配置项 | mock | staging | production |
|--------|------|---------|------------|
| `mode` | `mock` | `staging` | `production` |
| `client.server_url` | `http://127.0.0.1:8000` | `http://<公网IP>:8000` | `http://<公网IP>:8000` |
| `server.inference_url` | `http://localhost:8001` | `http://localhost:8001` | `http://localhost:8001` |
| `inference.device` | `cpu`(mock 不用) | `cuda` | `cuda` |
| 代码运行位置 | 全部本地 | 三台服务器 | 三台服务器 |
| 模型加载 | 不加载 | 真实加载 | 真实加载 |
| Milvus | 不需要 | Docker 部署 | Docker 部署 |
| SSH 隧道 | 不需要 | 需要 | 需要 |
