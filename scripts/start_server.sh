#!/bin/bash
# 启动 RAG Server (Public Server)
# 用法: bash scripts/start_server.sh [local|staging|production]
# 默认: local

MODE=${1:-local}
cd "$(dirname "$0")/.."
export NO_PROXY=localhost,127.0.0.1

echo "正在启动 RAG Server [模式: $MODE]..."
nohup python -m server.main --mode $MODE > logs/server.log 2>&1 &
echo "PID: $!"
echo "配置: server/config/${MODE}.yaml"
echo "日志: logs/server.log"
echo "监控: tail -f logs/server.log"
