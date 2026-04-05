#!/bin/bash
# 启动 RAG Server (Public Server)
# 用法: bash scripts/start_server.sh [local|staging|production]

MODE=$1
cd "$(dirname "$0")/.."
export NO_PROXY=localhost,127.0.0.1

echo "Starting RAG Server [mode: $MODE]..."
nohup python -m server.main --mode $MODE > logs/server.log 2>&1 &
echo "PID: $!"
echo "Config: server/config/${MODE}.yaml"
echo "Log:    logs/server.log"
echo "Watch:  tail -f logs/server.log"
