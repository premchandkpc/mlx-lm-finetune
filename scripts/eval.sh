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
MAX_SEQ="$(cfg max_seq_length "$CONFIG")"
MASK_PROMPT="$(cfg mask_prompt "$CONFIG")"
PROMPT="${PROMPT:-Hello! Introduce yourself in one sentence.}"

REPLY_LOSS=""
REPLY_PPL=""
PPL_STATUS=0

# Flags shared by both perplexity runs. mlx-lm never receives --config here
# (it would set train: true and start training), so every config value that
# moves the metric is forwarded explicitly — otherwise base and tuned are
# measured under different settings than the model was trained with:
#   --batch-size      must match the N < BATCH preflight below (mlx-lm
#                     otherwise falls back to its own default of 4)
#   --max-seq-length  otherwise mlx-lm truncates at its 2048 default
#   --mask-prompt     only when the config trains with it: the metric is
#                     assistant-reply-only, and mlx-lm raises "Prompt masking
#                     not supported for text dataset." if we force it there.
PPL_ARGS=(--model "$MODEL" --data "$DATA" --test --batch-size "$BATCH")
if [[ -n "$MAX_SEQ" ]]; then
  PPL_ARGS+=(--max-seq-length "$MAX_SEQ")
fi
case "$MASK_PROMPT" in
  True|true) PPL_ARGS+=(--mask-prompt) ;;
esac

# run_ppl <label> <args...> — evaluate test perplexity and print one line.
# Runs on $PPL_ARGS, so base and tuned share the config's batch size, sequence
# length and prompt masking. Returns non-zero when mlx-lm fails or the result
# cannot be parsed; callers record that in PPL_STATUS rather than pretending
# the comparison succeeded.
# Parses the result into REPLY_LOSS / REPLY_PPL (empty on parse failure).
run_ppl() {
  local label="$1"
  shift
  REPLY_LOSS=""
  REPLY_PPL=""
  local out line
  if ! out="$(.venv/bin/mlx_lm.lora "${PPL_ARGS[@]}" "$@" 2>&1)"; then
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
    run_ppl "base" --adapter-path "" || PPL_STATUS=1
    base_ppl="$REPLY_PPL"
    run_ppl "tuned" --adapter-path "$ADAPTER" || PPL_STATUS=1
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

# A failed perplexity run is a failed evaluation: report it instead of letting
# the script look successful with no numbers on screen.
if (( PPL_STATUS != 0 )); then
  echo "==> eval: perplexity run failed — no base/tuned comparison was produced (see output above)." >&2
fi

echo
echo "==> Sample generation with adapter: $ADAPTER"
GEN_STATUS=0
.venv/bin/mlx_lm.generate \
  --model "$MODEL" \
  --adapter-path "$ADAPTER" \
  --prompt "$PROMPT" \
  --max-tokens 200 \
  --temp 0.7 || GEN_STATUS=$?

if (( PPL_STATUS != 0 || GEN_STATUS != 0 )); then
  exit 1
fi
