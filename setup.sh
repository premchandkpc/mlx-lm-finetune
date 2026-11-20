#!/usr/bin/env bash
# Create a virtualenv and install MLX-LM fine-tuning dependencies.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

if [[ ! -d .venv ]]; then
  if command -v uv >/dev/null 2>&1; then
    echo "==> Creating venv with uv"
    uv venv .venv --python 3.12
  else
    echo "==> Creating venv with python3"
    python3 -m venv .venv
    .venv/bin/pip install --upgrade pip
  fi
else
  echo "==> .venv already exists, reusing it"
fi

# Install on every run: editing requirements.txt and re-running ./setup.sh has
# to take effect. uv-created venvs contain no pip, so prefer uv when present.
echo "==> Installing dependencies from requirements.txt"
if command -v uv >/dev/null 2>&1; then
  uv pip install --python .venv/bin/python -r requirements.txt
elif [[ -x .venv/bin/pip ]]; then
  .venv/bin/pip install -r requirements.txt
else
  echo "error: no uv on PATH and no pip inside .venv." >&2
  echo "  Install uv, or run: .venv/bin/python -m ensurepip --upgrade" >&2
  exit 1
fi

echo "==> Installed versions:"
.venv/bin/python -c "import mlx.core as mx; print('mlx:', mx.__version__)"
.venv/bin/python -c "import mlx_lm; print('mlx-lm:', getattr(mlx_lm, '__version__', 'unknown'))"
echo "Done. Activate with: source .venv/bin/activate"
