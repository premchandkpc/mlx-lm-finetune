#!/usr/bin/env bash
# Merge (fuse) the LoRA adapter into the base model weights, producing a
# standalone model that no longer needs --adapter-path at inference time.
#
# Usage:
#   ./scripts/fuse.sh
#   ./scripts/fuse.sh configs/lora.yaml
#   SAVE_PATH=my-fused-model ./scripts/fuse.sh
#   DEQUANTIZE=1 ./scripts/fuse.sh      # fp16 output instead of re-quantized 4-bit
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

CONFIG="${1:-configs/lora.yaml}"
if [[ $# -gt 0 ]]; then shift; fi

RESOLVED="$(RESOLVE_QUIET=1 resolve_cfg "$CONFIG")"
EXTRA=()
[[ -n "${DEQUANTIZE:-}" ]] && EXTRA+=(--dequantize)

exec .venv/bin/mlx_lm.fuse \
  --model "$(cfg model "$RESOLVED")" \
  --adapter-path "$(cfg adapter_path "$RESOLVED")" \
  --save-path "${SAVE_PATH:-fused_model}" \
  ${EXTRA[@]+"${EXTRA[@]}"} \
  "$@"
