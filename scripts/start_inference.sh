#!/bin/bash
# 启动 Inference Worker (GPU Server)
# 用法: bash scripts/start_inference.sh

cd "$(dirname "$0")/.."
export NO_PROXY=localhost,127.0.0.1

echo "正在启动 Inference Worker..."
nohup python -m inference.main > logs/inference.log 2>&1 &
echo "PID: $!"
echo "日志: logs/inference.log"
echo "监控: tail -f logs/inference.log"
