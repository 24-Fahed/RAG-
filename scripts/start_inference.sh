#!/bin/bash
# 启动 Inference Worker (GPU Server)
# 用法: bash scripts/start_inference.sh [local|staging|production]
# 默认: local

MODE=${1:-local}
cd "$(dirname "$0")/.."
export NO_PROXY=localhost,127.0.0.1

echo "正在启动 Inference Worker [模式: $MODE]..."
nohup python -m inference.main --mode $MODE > logs/inference.log 2>&1 &
echo "PID: $!"
echo "配置: inference/config/${MODE}.yaml"
echo "日志: logs/inference.log"
echo "监控: tail -f logs/inference.log"
