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

所有配置集中在项目根目录的 `deploy.yaml` 中，支持三种模式：

| 模式 | 说明 |
|------|------|
| `mock` | 本地测试，无需 GPU，推理返回模拟数据。 |
| `staging` | 部署到真实服务器，用于集成测试和参数调优。 |
| `production` | 与 staging 相同的服务器，正式上线版本。 |

通过编辑 `deploy.yaml` 切换模式：

```yaml
mode: mock   # mock | staging | production
```

## 快速开始（Mock 模式 - 本地测试）

Mock 模式可以在本地无 GPU 和真实模型的情况下验证整个流程。

### 1. 启动 Inference Worker（mock）

```bash
cd rag_deploy
# deploy.yaml 中确认 mode: mock
python -m inference.main
```

### 2. 启动 RAG Server

```bash
# 另一个终端
python -m server.main
```

### 3. 使用 Client

```bash
python client/client.py query "What is RAG?"
```

## 生产部署

完整的三阶段部署说明请参见 [DEPLOY_GUIDE.md](DEPLOY_GUIDE.md)。

## 非侵入式设计

`rag_deploy/` 项目**不修改** `rag_langgraph/` 中的任何代码，而是通过导入和封装现有模块实现：

- `inference/` 从 `rag_langgraph.models.*` 导入（classifier, generator, rerankers, compressors）
- `inference/` 从 `rag_langgraph.indexing.embedding` 导入
- `server/` 从 `rag_langgraph.indexing.vectorstore` 和 `rag_langgraph.indexing.*` 导入
