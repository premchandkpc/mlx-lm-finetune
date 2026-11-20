#!/usr/bin/env bash
# Shared helpers for the scripts/ commands. Source this, don't run it.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ ! -x .venv/bin/mlx_lm.lora ]]; then
  echo "error: virtualenv not set up. Run ./setup.sh first." >&2
  exit 1
fi

# cfg <key> [config-file] — read a top-level value from the YAML config
# (resolves `extends:` and parses 1e-4 as a float, via scripts/cfgtools.py).
cfg() {
  local key="$1" file="${2:-configs/lora.yaml}"
  .venv/bin/python - "$file" "$key" <<'PY'
import sys
sys.path.insert(0, "scripts")
import cfgtools
val = cfgtools.load_config(sys.argv[1]).get(sys.argv[2], "")
print("" if val is None else val)
PY
}

# resolve_cfg <config-file> [cli-args...] — write a merged copy of the config
# (extends resolved, LR schedule re-synced with --iters/--learning-rate) under
# outputs/.resolved/ and print its path. Preflight warnings go to stderr
# (silence with RESOLVE_QUIET=1).
resolve_cfg() {
  local config="$1"
  shift
  mkdir -p outputs/.resolved
  local quiet=()
  [[ -n "${RESOLVE_QUIET:-}" ]] && quiet=(--quiet)
  # ${arr[@]+...} keeps empty arrays safe under `set -u` on macOS's bash 3.2.
  .venv/bin/python scripts/resolve_config.py "$config" \
    --out "outputs/.resolved/$(basename "$config")" ${quiet[@]+"${quiet[@]}"} -- "$@"
}

# effective_batch <config-file> [cli-args...] — resolve batch size the same
# way mlx-lm does: CLI --batch-size wins, then config batch_size, then 4.
effective_batch() {
  local config="$1"
  shift
  .venv/bin/python - "$config" "$@" <<'PY'
import sys
sys.path.insert(0, "scripts")
import cfgtools
args = sys.argv[1:]
config, cli = args[0], args[1:]
batch = None
for i, a in enumerate(cli):
    if a == "--batch-size" and i + 1 < len(cli):
        batch = int(cli[i + 1])
    elif a.startswith("--batch-size="):
        batch = int(a.split("=", 1)[1])
if batch is None:
    batch = cfgtools.load_config(config).get("batch_size") or 4
print(batch)
PY
}

# require_split <label> <path> <batch> <required|optional>
# mlx-lm raises "Dataset must have at least batch_size examples" at runtime;
# fail earlier with an actionable message instead.
require_split() {
  local label="$1" path="$2" batch="$3" mode="$4"
  if [[ ! -f "$path" ]]; then
    if [[ "$mode" == "required" ]]; then
      echo "error: $label not found: $path" >&2
      echo "  Run: make data   (or scripts/prepare_data.py --input <your-data>)" >&2
      exit 1
    fi
    return 0
  fi
  local n
  n="$(wc -l < "$path" | tr -d '[:space:]')"
  if (( n < batch )); then
    echo "error: $label ($path) has $n example(s) but batch_size=$batch." >&2
    echo "  mlx-lm needs at least batch_size examples in every split it reads." >&2
    echo "  Fix one of:" >&2
    echo "    - re-run scripts/prepare_data.py with --min-split $batch" >&2
    echo "    - pass --batch-size $n (or smaller) to the training command" >&2
    echo "    - lower --valid-split / --test-split or add more data" >&2
    exit 1
  fi
}
