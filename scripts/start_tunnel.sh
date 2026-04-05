#!/bin/bash
# 建立 SSH 隧道到 GPU Server (Public Server 上执行)
# 用法: bash scripts/start_tunnel.sh
# 修改下方 GPU_PORT 为你的 AutoDL SSH 端口

GPU_PORT=<gpu_ssh_port>
GPU_HOST=root@connect.westb.seetacloud.com

echo "正在建立 SSH 隧道 -> localhost:8001 -> GPU:8000 ..."
echo "配置: server/config/staging.yaml (inference_url: http://localhost:8001)"
autossh -M 0 -N -L 8001:localhost:8000 -p $GPU_PORT $GPU_HOST \
    -o ServerAliveInterval=60 -o ServerAliveCountMax=3
