#!/bin/bash
# 启动 RAG Client (User PC)
# 用法: bash scripts/start_client.sh [local|staging|production] [query|index] [args...]

MODE=$1
cd "$(dirname "$0")/.."
shift

echo "RAG Client [mode: $MODE] [config: client/config/${MODE}.yaml]"
python -m client.client --mode $MODE "$@"
