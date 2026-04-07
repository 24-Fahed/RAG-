#!/bin/bash
# 建立 SSH 隧道到 GPU Server（在公网服务器上执行）
# 用法: bash scripts/start_tunnel.sh
# 修改下方 GPU_PORT 和 GPU_HOST 为你的实际 SSH 信息

GPU_PORT=<gpu_ssh_port>
GPU_HOST=root@<gpu_ssh_host>

echo "Building SSH tunnel -> localhost:8001 -> GPU:8000 ..."
echo "Config: server/config/staging.yaml (inference_url: http://localhost:8001)"
autossh -M 0 -N -L 8001:localhost:8000 -p $GPU_PORT $GPU_HOST \
    -o ServerAliveInterval=60 -o ServerAliveCountMax=3
