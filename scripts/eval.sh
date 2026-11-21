#!/usr/bin/env bash
# Did fine-tuning help? Compare base vs base+adapter on held-out data, then
# print one sample generation.
#
#   1. scripts/compare.py: loss/perplexity (same masking as training), greedy
#      generation F1 / exact-match vs the reference answers, side-by-side samples.
#      Report: outputs/compare/report.md
#   2. one sample generation with the adapter loaded
#
# Usage:
#   ./scripts/eval.sh
#   ./scripts/eval.sh configs/lora.yaml
#   ./scripts/eval.sh configs/lora.yaml --split valid --gen-limit 50   # -> compare.py
#   PROMPT="Explain MLX in one sentence." ./scripts/eval.sh
#   GEN_ARGS="--top-p 0.9" ./scripts/eval.sh                           # -> mlx_lm.generate
#
# Extra CLI args go to compare.py only; mlx_lm.generate has a different flag
# set (e.g. no --batch-size), so it gets GEN_ARGS instead.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

CONFIG="${1:-configs/lora.yaml}"
if [[ $# -gt 0 ]]; then shift; fi

RESOLVED="$(RESOLVE_QUIET=1 resolve_cfg "$CONFIG")"
MODEL="$(cfg model "$RESOLVED")"
ADAPTER="$(cfg adapter_path "$RESOLVED")"
PROMPT="${PROMPT:-Hello! Introduce yourself in one sentence.}"

echo "==> Base vs tuned (held-out)"
.venv/bin/python scripts/compare.py --config "$RESOLVED" "$@"

echo
echo "==> Sample generation with adapter: $ADAPTER"
read -r -a GEN <<< "${GEN_ARGS:-}"
.venv/bin/mlx_lm.generate \
  --model "$MODEL" \
  --adapter-path "$ADAPTER" \
  --prompt "$PROMPT" \
  --max-tokens 200 \
  --temp 0.7 ${GEN[@]+"${GEN[@]}"}
