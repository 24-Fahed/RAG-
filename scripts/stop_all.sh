#!/bin/bash
# 停止所有 RAG 服务
# 用法: bash scripts/stop_all.sh

echo "Stopping Inference Worker..."
pkill -f "python -m inference.main" 2>/dev/null && echo "  Stopped" || echo "  Not running"

echo "Stopping RAG Server..."
pkill -f "python -m server.main" 2>/dev/null && echo "  Stopped" || echo "  Not running"

echo "Stopping SSH tunnel..."
pkill -f "ssh -N -L 8001" 2>/dev/null && echo "  Stopped" || echo "  Not running"
