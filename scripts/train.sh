#!/usr/bin/env bash
# Fine-tune an SLM with LoRA/DoRA/full tuning via mlx-lm.
#
# Usage:
#   ./scripts/train.sh                          # use configs/lora.yaml
#   ./scripts/train.sh configs/lora.yaml        # explicit config
#   ./scripts/train.sh configs/lora.yaml --iters 100 --batch-size 2
#   # any mlx_lm.lora flag can override config values:
#   ./scripts/train.sh --model mlx-community/SmolLM2-360M-Instruct-4bit
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

CONFIG="configs/lora.yaml"
if [[ $# -gt 0 && "${1}" != --* ]]; then
  CONFIG="$1"
  shift
fi

DATA="$(cfg data "$CONFIG")"
BATCH="$(effective_batch "$CONFIG" "$@")"

echo "==> Config: $CONFIG"
echo "==> Model : $(cfg model "$CONFIG")"
echo "==> Data  : $DATA"
echo "==> Batch : $BATCH"

# Preflight: local datasets must have >= batch_size examples per split.
# (Skip silently when `data:` is a Hugging Face dataset name.)
if [[ -d "$DATA" ]]; then
  require_split "train.jsonl" "$DATA/train.jsonl" "$BATCH" required
  require_split "valid.jsonl" "$DATA/valid.jsonl" "$BATCH" optional
  require_split "test.jsonl"  "$DATA/test.jsonl"  "$BATCH" optional
fi

exec .venv/bin/mlx_lm.lora --config "$CONFIG" "$@"
