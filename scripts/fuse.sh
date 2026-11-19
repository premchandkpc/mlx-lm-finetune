#!/usr/bin/env bash
# Merge (fuse) the LoRA adapter into the base model weights, producing a
# standalone model that no longer needs --adapter-path at inference time.
#
# Usage:
#   ./scripts/fuse.sh
#   ./scripts/fuse.sh configs/lora.yaml
#   SAVE_PATH=my-fused-model ./scripts/fuse.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

CONFIG="${1:-configs/lora.yaml}"
if [[ $# -gt 0 ]]; then shift; fi

exec .venv/bin/mlx_lm.fuse \
  --model "$(cfg model "$CONFIG")" \
  --adapter-path "$(cfg adapter_path "$CONFIG")" \
  --save-path "${SAVE_PATH:-fused_model}" \
  "$@"
