#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

TRAIN_TOKENS_PATH="${TRAIN_TOKENS_PATH:-data/tinystories/train_tokens.npy}"
VALID_TOKENS_PATH="${VALID_TOKENS_PATH:-data/tinystories/valid_tokens.npy}"

# 常改参数（建议只改这几项）
BATCH_SIZE="${BATCH_SIZE:-32}"
TOTAL_TOKENS="${TOTAL_TOKENS:-327680000}"
MAX_LEARNING_RATE="${MAX_LEARNING_RATE:-3e-4}"
USE_WANDB="${USE_WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-cs336-assignment1}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-tinystories-17m-once}"
WANDB_GROUP="${WANDB_GROUP:-tinystories-17m}"
DEVICE="${DEVICE:-auto}"

# 作业 17M 预设（固定）
VOCAB_SIZE="10000"
CONTEXT_LENGTH="256"
D_MODEL="512"
NUM_LAYERS="4"
NUM_HEADS="16"
D_FF="1344"
ROPE_THETA="10000"

MAX_ITERS="${MAX_ITERS:-$((TOTAL_TOKENS / (BATCH_SIZE * CONTEXT_LENGTH)))}"
MIN_LEARNING_RATE="3e-5"
WARMUP_ITERS="400"
COSINE_CYCLE_ITERS="${COSINE_CYCLE_ITERS:-$MAX_ITERS}"
BETA1="0.9"
BETA2="0.95"
EPS="1e-8"
WEIGHT_DECAY="0.1"
GRAD_CLIP_NORM="1.0"

EVAL_INTERVAL="100"
EVAL_BATCHES="20"
LOG_INTERVAL="10"

CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints/tinystories_17m_once}"
CHECKPOINT_INTERVAL="500"
SEED="42"

if [[ ! -f "$TRAIN_TOKENS_PATH" ]]; then
  echo "训练集 tokens 文件不存在: $TRAIN_TOKENS_PATH"
  exit 1
fi

if [[ ! -f "$VALID_TOKENS_PATH" ]]; then
  echo "验证集 tokens 文件不存在: $VALID_TOKENS_PATH"
  exit 1
fi

CMD=(
  uv run python -m cs336_basics.train
  --train-tokens-path "$TRAIN_TOKENS_PATH"
  --valid-tokens-path "$VALID_TOKENS_PATH"
  --vocab-size "$VOCAB_SIZE"
  --context-length "$CONTEXT_LENGTH"
  --d-model "$D_MODEL"
  --num-layers "$NUM_LAYERS"
  --num-heads "$NUM_HEADS"
  --d-ff "$D_FF"
  --rope-theta "$ROPE_THETA"
  --batch-size "$BATCH_SIZE"
  --max-iters "$MAX_ITERS"
  --max-learning-rate "$MAX_LEARNING_RATE"
  --min-learning-rate "$MIN_LEARNING_RATE"
  --warmup-iters "$WARMUP_ITERS"
  --cosine-cycle-iters "$COSINE_CYCLE_ITERS"
  --beta1 "$BETA1"
  --beta2 "$BETA2"
  --eps "$EPS"
  --weight-decay "$WEIGHT_DECAY"
  --grad-clip-norm "$GRAD_CLIP_NORM"
  --eval-interval "$EVAL_INTERVAL"
  --eval-batches "$EVAL_BATCHES"
  --log-interval "$LOG_INTERVAL"
  --checkpoint-dir "$CHECKPOINT_DIR"
  --checkpoint-interval "$CHECKPOINT_INTERVAL"
  --device "$DEVICE"
  --seed "$SEED"
)

if [[ "$USE_WANDB" == "1" ]]; then
  CMD+=(
    --use-wandb
    --wandb-project "$WANDB_PROJECT"
    --wandb-run-name "$WANDB_RUN_NAME"
    --wandb-group "$WANDB_GROUP"
  )
  if [[ -n "$WANDB_ENTITY" ]]; then
    CMD+=(--wandb-entity "$WANDB_ENTITY")
  fi
fi

"${CMD[@]}"
