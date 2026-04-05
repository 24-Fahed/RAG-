#!/bin/bash
# 启动 RAG Client (User PC)
# 用法: bash scripts/start_client.sh [local|staging|production] [query|index] [参数...]
# 默认: local

MODE=${1:-local}
cd "$(dirname "$0")/.."
shift

echo "RAG Client [模式: $MODE] [配置: client/config/${MODE}.yaml]"
python -m client.client --mode $MODE "$@"
