# RAG 部署 - 分布式 RAG 系统

基于 [FudanDNN-NLP/RAG](https://github.com/FudanDNN-NLP/RAG)，使用 LangGraph 重建并分布式部署到三台机器。

## 架构

三张架构图见 `docs/architecture/` 目录：

| 架构图 | 文件 |
|--------|------|
| 逻辑架构 | `docs/architecture/RAG_Logical_Architecture.png` |
| 程序单元 | `docs/architecture/RAG_Program_Units.png` |
| 物理架构 | `docs/architecture/RAG_Physical_Architecture.png` |

## 三个程序单元

| 单元 | 目录 | 部署目标 | 说明 |
|------|------|----------|------|
| PU-1: Client | `client/` | 用户 PC | 用于查询和文档提交的 CLI 工具 |
| PU-2: RAG Server | `server/` | 公网服务器 | FastAPI 网关 + Milvus + BM25 + 流程编排 |
| PU-3: Inference Worker | `inference/` | GPU 服务器 (AutoDL) | 所有 ML 模型推理（LLM, BERT, BGE, MonoT5, Recomp） |

## 配置

每个程序单元有独立的配置目录（`config/`），包含三个版本的 YAML 配置：

| 程序单元 | 配置目录 | 配置文件 |
|----------|----------|----------|
| PU-1 Client | `client/config/` | `local.yaml`, `staging.yaml`, `production.yaml` |
| PU-2 RAG Server | `server/config/` | `local.yaml`, `staging.yaml`, `production.yaml` |
| PU-3 Inference Worker | `inference/config/` | `local.yaml`, `staging.yaml`, `production.yaml` |

通过 `--mode` 参数选择配置版本：

```bash
python -m inference.main --mode local       # 使用 inference/config/local.yaml
python -m server.main --mode staging        # 使用 server/config/staging.yaml
python -m client.client --mode production query "What is RAG?"
```

或通过启动脚本：

```bash
bash scripts/start_inference.sh staging
bash scripts/start_server.sh production
bash scripts/start_client.sh local query "What is RAG?"
```

## 快速开始（本地测试）

本地测试无需 GPU 和真实模型。

### 1. 启动 Inference Worker（mock）

```bash
cd rag_deploy
python -m inference.main --mode local
```

### 2. 启动 RAG Server

```bash
# 另一个终端
python -m server.main --mode local
```

### 3. 使用 Client

```bash
python -m client.client --mode local query "What is RAG?"
```

## 生产部署

完整的三阶段部署说明请参见 [DEPLOY_GUIDE.md](DEPLOY_GUIDE.md)。

## 临时补丁

| 补丁 | 文件 | 说明 |
|------|------|------|
| transformers 版本固定 | `inference/requirements.txt` | 固定到 `transformers==4.48.2`，与 `torch 2.5.1` 兼容，避免 5.x 的 torch>=2.6 限制 |
| HuggingFace 镜像 | `inference/main.py` | GPU 服务器无法访问 huggingface.co，走 hf-mirror.com |
| 跳过分类步骤 | `server/services/pipeline.py` | 分类器未微调，临时强制所有查询走检索 |

## 非侵入式设计

`rag_deploy/` 项目**不修改** `rag_langgraph/` 中的任何代码，而是通过导入和封装现有模块实现：

- `inference/` 从 `rag_langgraph.models.*` 导入（classifier, generator, rerankers, compressors）
- `inference/` 从 `rag_langgraph.indexing.embedding` 导入
- `server/` 从 `rag_langgraph.indexing.vectorstore` 和 `rag_langgraph.indexing.*` 导入
