#!/usr/bin/env bash
# Fine-tune an SLM with LoRA/DoRA/full tuning via mlx-lm.
#
# Usage:
#   ./scripts/train.sh                          # use configs/lora.yaml
#   ./scripts/train.sh configs/quality.yaml     # a preset
#   ./scripts/train.sh configs/lora.yaml --iters 100 --batch-size 2
#   # any mlx_lm.lora flag can override config values:
#   ./scripts/train.sh --model mlx-community/SmolLM2-360M-Instruct-4bit
#
# The config is resolved first (extends merged; lr_schedule re-synced with
# --iters / --learning-rate) and preflight warnings are printed, e.g. when the
# run would loop ~200 epochs over a tiny train set.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

CONFIG="configs/lora.yaml"
if [[ $# -gt 0 && "${1}" != --* ]]; then
  CONFIG="$1"
  shift
fi

RESOLVED="$(resolve_cfg "$CONFIG" "$@")"
DATA="$(cfg data "$RESOLVED")"
BATCH="$(effective_batch "$RESOLVED" "$@")"

echo "==> Config: $CONFIG (resolved: $RESOLVED)"
echo "==> Model : $(cfg model "$RESOLVED")"
echo "==> Data  : $DATA"
echo "==> Batch : $BATCH"

# Preflight: local datasets must have >= batch_size examples per split.
# (Skip silently when `data:` is a Hugging Face dataset name.)
if [[ -d "$DATA" ]]; then
  require_split "train.jsonl" "$DATA/train.jsonl" "$BATCH" required
  require_split "valid.jsonl" "$DATA/valid.jsonl" "$BATCH" optional
  require_split "test.jsonl"  "$DATA/test.jsonl"  "$BATCH" optional
fi

exec .venv/bin/mlx_lm.lora --config "$RESOLVED" "$@"
