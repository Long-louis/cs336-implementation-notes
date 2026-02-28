#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 4 ]]; then
  echo "用法: $0 <gpu_host> <remote_dir> [--with-data] [--delete]"
  echo "示例: $0 user@gpu-server /workspace/cs336 --with-data"
  exit 1
fi

GPU_HOST="$1"
REMOTE_DIR="$2"
WITH_DATA="false"
DELETE_MODE="false"

for arg in "${@:3}"; do
  if [[ "$arg" == "--with-data" ]]; then
    WITH_DATA="true"
  elif [[ "$arg" == "--delete" ]]; then
    DELETE_MODE="true"
  else
    echo "未知参数: $arg"
    exit 1
  fi
done

LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"

RSYNC_ARGS=(
  -azP
  --exclude ".git/"
  --exclude ".venv/"
  --exclude "**/__pycache__/"
  --exclude "**/.pytest_cache/"
  --exclude "**/.mypy_cache/"
  --exclude "**/.ruff_cache/"
  --exclude "**/wandb/"
  --exclude "**/checkpoints/"
)

if [[ "$WITH_DATA" != "true" ]]; then
  RSYNC_ARGS+=(--exclude "assignment1-basics/data/")
fi

if [[ "$DELETE_MODE" == "true" ]]; then
  RSYNC_ARGS+=(--delete)
fi

echo "开始同步到 GPU: ${GPU_HOST}:${REMOTE_DIR}"
rsync "${RSYNC_ARGS[@]}" "${LOCAL_DIR}/" "${GPU_HOST}:${REMOTE_DIR}/"
echo "同步完成"
