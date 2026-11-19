#!/usr/bin/env bash
# Evaluate the fine-tuned adapter:
#   1. test-set perplexity (if data/test.jsonl exists)
#   2. a short sample generation so you can eyeball the output
#
# Usage:
#   ./scripts/eval.sh
#   ./scripts/eval.sh configs/lora.yaml
#   PROMPT="Explain MLX in one sentence." ./scripts/eval.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

CONFIG="${1:-configs/lora.yaml}"
if [[ $# -gt 0 ]]; then shift; fi

MODEL="$(cfg model "$CONFIG")"
DATA="$(cfg data "$CONFIG")"
ADAPTER="$(cfg adapter_path "$CONFIG")"
BATCH="$(effective_batch "$CONFIG" "$@")"
PROMPT="${PROMPT:-Hello! Introduce yourself in one sentence.}"

if [[ -f "$DATA/test.jsonl" ]]; then
  N="$(wc -l < "$DATA/test.jsonl" | tr -d '[:space:]')"
  if (( N < BATCH )); then
    echo "==> Skipping perplexity: test.jsonl has $N example(s) < batch_size=$BATCH."
    echo "    Re-run prepare_data.py with --min-split $BATCH to fix it."
  else
    echo "==> Test-set perplexity ($DATA/test.jsonl)"
    .venv/bin/mlx_lm.lora \
      --model "$MODEL" \
      --adapter-path "$ADAPTER" \
      --data "$DATA" \
      --test "$@"
  fi
else
  echo "==> Skipping perplexity: $DATA/test.jsonl not found."
  echo "    Re-run prepare_data.py with --test-split 0.05 to create it."
fi

echo
echo "==> Sample generation with adapter: $ADAPTER"
.venv/bin/mlx_lm.generate \
  --model "$MODEL" \
  --adapter-path "$ADAPTER" \
  --prompt "$PROMPT" \
  --max-tokens 200 \
  --temp 0.7 "$@"
