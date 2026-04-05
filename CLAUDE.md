# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Distributed RAG (Retrieval-Augmented Generation) system based on [FudanDNN-NLP/RAG](https://github.com/FudanDNN-NLP/RAG), rebuilt with LangGraph and deployed across three machines. Licensed under Apache-2.0.

## Architecture

Three Program Units (PU), each independently deployed:

- **PU-1 Client** (`client/`) → User PC. CLI tool for queries and document upload.
- **PU-2 RAG Server** (`server/`) → Public server. FastAPI gateway, Milvus (Docker), BM25 (local files), pipeline orchestration.
- **PU-3 Inference Worker** (`inference/`) → GPU server (AutoDL). All ML model inference.

Communication: Client →(HTTP)→ RAG Server →(SSH tunnel)→ Inference Worker. The GPU server has no public IP; the public server opens an SSH tunnel (`ssh -N -L 8001:localhost:8000`) to reach it.

`rag_langgraph/` contains the core RAG library (models, nodes, graphs, indexing). The deployment layer (`server/`, `inference/`) wraps it without modifying it.

## Configuration System

Each PU has its own `config/` directory with three YAML files, selected via `--mode` CLI argument:

| PU | Config Dir | Files |
|----|-----------|-------|
| PU-1 Client | `client/config/` | `local.yaml`, `staging.yaml`, `production.yaml` |
| PU-2 RAG Server | `server/config/` | `local.yaml`, `staging.yaml`, `production.yaml` |
| PU-3 Inference Worker | `inference/config/` | `local.yaml`, `staging.yaml`, `production.yaml` |

**How it works:** Each PU's `config.py` has a `load_config(mode)` function that loads `config/{mode}.yaml` from its own directory. Each entry point (`server/main.py`, `inference/main.py`, `client/client.py`) parses `--mode` with argparse **before** importing config-dependent modules. The config is lazily loaded and cached.

**Key config fields:**
- `mode`: `mock` / `staging` / `production` — controls whether inference loads mock routers or real models
- `client.server_url`: URL the client uses to reach RAG Server
- `server.inference_url`: URL RAG Server uses to reach Inference Worker (via SSH tunnel in staging/prod)
- `inference.device`: `cpu` or `cuda`

## Running

```bash
# Local mock test (no GPU needed)
python -m inference.main --mode local    # terminal 1, mock inference on :8000
python -m server.main --mode local       # terminal 2, RAG server on :8001
python -m client.client --mode local query "What is RAG?"  # terminal 3

# On Linux servers (via scripts)
bash scripts/start_inference.sh staging
bash scripts/start_server.sh production
bash scripts/start_client.sh local query "What is RAG?"
bash scripts/start_tunnel.sh    # SSH tunnel to GPU server
bash scripts/stop_all.sh        # stop everything
```

**Note on uvicorn reload:** `server/main.py` uses `reload=True` for local/staging, `reload=False` for production. The `--mode` argument is re-parsed from `sys.argv` on reload.

## RAG Pipeline (query flow)

`server/services/pipeline.py` `run_query_pipeline()`:

```
classify(GPU) → hyde(GPU) → embed(GPU) → milvus_search(local)
→ bm25_search(local) → hybrid_fuse(local) → rerank(GPU)
→ repack(local) → compress(GPU) → generate(GPU)
```

GPU tasks are delegated to Inference Worker via `server/services/inference_client.py` (httpx client pointing to `inference_url`).

Hybrid fusion formula: `score = alpha * sparse + (1 - alpha) * dense` (default alpha=0.3).

## Key Models (all in rag_langgraph/models/)

| Model | Purpose | ~VRAM |
|-------|---------|-------|
| LLaMA-3-8B-Instruct | Generation, HyDE | 16GB |
| BERT multilingual | Classification (need retrieval?) | 700MB |
| BGE-base-en-v1.5 | Embedding (768-dim) | 400MB |
| MonoT5-base | Reranking | 1GB |
| Recomp extractive | Context compression | 400MB |
| BGE-reranker-v2-m3 | Alternative reranker | 1.2GB |

Models auto-download from HuggingFace on first use (except LLM which must be pre-downloaded).

## Milvus

Deployed via `docker/docker-compose.yml` (etcd + minio + milvus-standalone). Default port 19530.

If the public server uses a proxy, set `NO_PROXY=localhost,127.0.0.1` or Milvus connections will fail.

## Language

All comments, docstrings, and documentation are in Chinese. Technical terms (FastAPI, Milvus, BM25, etc.) stay in English. PUML diagram text must be in English to avoid garbled characters in generated PNGs. Use `java -jar plantuml.jar` to generate PNGs from PUML files.
