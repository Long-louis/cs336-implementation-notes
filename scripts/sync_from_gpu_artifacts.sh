#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "用法: $0 <gpu_host> <remote_dir>"
  echo "示例: $0 user@gpu-server /workspace/cs336"
  exit 1
fi

GPU_HOST="$1"
REMOTE_DIR="$2"
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "开始从 GPU 拉取训练产物"
rsync -azP \
  --prune-empty-dirs \
  --include "*/" \
  --include "**/checkpoints/***" \
  --include "**/wandb/***" \
  --include "**/runs/***" \
  --exclude "*" \
  "${GPU_HOST}:${REMOTE_DIR}/" "${LOCAL_DIR}/"

echo "拉取完成"
