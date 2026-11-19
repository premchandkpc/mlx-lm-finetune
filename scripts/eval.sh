#!/usr/bin/env bash
# Evaluate the fine-tuned adapter:
#   1. test-set perplexity for BOTH the base model and the tuned model,
#      printed side by side with the relative change — the number that
#      answers "did fine-tuning help?"
#   2. a short sample generation so you can eyeball the output
#
# Usage:
#   ./scripts/eval.sh                          # use configs/lora.yaml
#   ./scripts/eval.sh configs/lora.yaml
#   PROMPT="Your question here" ./scripts/eval.sh
#
# No training/generation flags are accepted: `mlx_lm.lora --test` and
# `mlx_lm.generate` take different options, so forwarding "$@" to both
# would break one of them. Change behavior via the config or PROMPT.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

CONFIG="configs/lora.yaml"
if [[ $# -gt 0 ]]; then
  CONFIG="$1"
  shift
fi
if [[ $# -gt 0 ]]; then
  echo "usage: $0 [config-file]   (set PROMPT=... to change the sample prompt)" >&2
  exit 2
fi

MODEL="$(cfg model "$CONFIG")"
DATA="$(cfg data "$CONFIG")"
ADAPTER="$(cfg adapter_path "$CONFIG")"
BATCH="$(effective_batch "$CONFIG")"
PROMPT="${PROMPT:-Hello! Introduce yourself in one sentence.}"

REPLY_LOSS=""
REPLY_PPL=""

# run_ppl <label> <args...> — evaluate test perplexity and print one line.
# Both runs pass --mask-prompt so the metric is assistant-reply-only: the
# task is predicting the reply, and prompt tokens are trivially predictable,
# which would hide regressions. (We cannot pass --config here: it sets
# train: true and would start training instead of testing.)
# Parses the result into REPLY_LOSS / REPLY_PPL (empty on parse failure).
run_ppl() {
  local label="$1"
  shift
  REPLY_LOSS=""
  REPLY_PPL=""
  local out line
  if ! out="$(.venv/bin/mlx_lm.lora --model "$MODEL" --data "$DATA" --test --mask-prompt "$@" 2>&1)"; then
    printf '%s\n' "$out" >&2
    return 1
  fi
  line="$(printf '%s\n' "$out" | tr '\r' '\n' \
    | sed 's/\x1b\[[0-9;]*m//g' \
    | grep -E 'Test loss' | tail -1 || true)"
  if [[ -z "$line" ]]; then
    echo "==> could not parse perplexity output, raw log follows:" >&2
    printf '%s\n' "$out" >&2
    return 1
  fi
  printf '  %-5s %s\n' "$label" "$line"
  REPLY_LOSS="$(sed -n 's/.*Test loss \([0-9.eE+-]*\),.*/\1/p' <<<"$line")"
  REPLY_PPL="$(sed -n 's/.*Test ppl \([0-9.eE+-]*\).*/\1/p' <<<"$line")"
}

if [[ -f "$DATA/test.jsonl" ]]; then
  N="$(wc -l < "$DATA/test.jsonl" | tr -d '[:space:]')"
  if (( N < BATCH )); then
    echo "==> Skipping perplexity: test.jsonl has $N example(s) < batch_size=$BATCH."
    echo "    Re-run prepare_data.py with --min-split $BATCH to fix it."
  else
    echo "==> Test perplexity ($DATA/test.jsonl)"
    run_ppl "base" --adapter-path "" || true
    base_ppl="$REPLY_PPL"
    run_ppl "tuned" --adapter-path "$ADAPTER" || true
    tuned_ppl="$REPLY_PPL"
    if [[ -n "$base_ppl" && -n "$tuned_ppl" ]]; then
      awk -v b="$base_ppl" -v t="$tuned_ppl" 'BEGIN {
        printf "  delta  %+.1f%% perplexity (tuned vs base; lower is better)\n",
               (t - b) / b * 100
      }'
    fi
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
  --temp 0.7
