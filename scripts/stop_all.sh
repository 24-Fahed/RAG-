#!/bin/bash
# 停止所有 RAG 服务
# 用法: bash scripts/stop_all.sh

echo "正在停止 Inference Worker..."
pkill -f "python -m inference.main" 2>/dev/null && echo "  Inference Worker 已停止" || echo "  未在运行"

echo "正在停止 RAG Server..."
pkill -f "python -m server.main" 2>/dev/null && echo "  RAG Server 已停止" || echo "  未在运行"

echo "正在停止 SSH 隧道..."
pkill -f "ssh -N -L 8001" 2>/dev/null && echo "  SSH 隧道已停止" || echo "  未在运行"
