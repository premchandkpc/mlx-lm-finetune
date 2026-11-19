#!/usr/bin/env bash
# Interactive chat with the fine-tuned model.
#
# Usage:
#   ./scripts/chat.sh
#   ./scripts/chat.sh configs/lora.yaml
#   SYSTEM="You are a terse technical assistant." ./scripts/chat.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

CONFIG="${1:-configs/lora.yaml}"
if [[ $# -gt 0 ]]; then shift; fi

ARGS=(--model "$(cfg model "$CONFIG")" --adapter-path "$(cfg adapter_path "$CONFIG")")
[[ -n "${SYSTEM:-}" ]] && ARGS+=(--system-prompt "$SYSTEM")

exec .venv/bin/mlx_lm.chat "${ARGS[@]}" "$@"
